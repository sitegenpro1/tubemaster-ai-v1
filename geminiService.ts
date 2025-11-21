
import { ThumbnailGenResult } from "../types";

/**
 * SERVICE ARCHITECTURE:
 * 1. THE BRAIN: Groq API (Moonshot Kimi) - Handles all text, logic, strategy, and JSON parsing.
 * 2. IMAGE GEN: Pollinations.ai (Flux Model) - Handles generating new images (unlimited).
 * 3. VISUAL EYES: OpenRouter (Grok 4.1 Fast) or Groq (Llama 3.2 Vision)
 */

const GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions";
const OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions";

// Models
const MODEL_BRAIN = "moonshotai/kimi-k2-instruct-0905"; // Excellent Logic/Text Model
const MODEL_VISION_GROQ = "llama-3.2-90b-vision-preview"; // Required for Groq Vision. Kimi is Text-Only.

// UPDATED: Using x-ai/grok-4.1-fast as requested
const MODEL_VISION_FREE_PRIMARY = "x-ai/grok-4.1-fast"; // Primary: High Speed Vision
const MODEL_VISION_FREE_FALLBACK = "meta-llama/llama-3.2-11b-vision-instruct:free"; // Backup: Reliable

const getApiKey = (): string => {
  // 1. Check Import Meta (Vite native)
  const meta = import.meta as any;
  if (meta.env && meta.env.VITE_GROQ_API_KEY) return meta.env.VITE_GROQ_API_KEY;
  
  // 2. Check Process Env (Injected by vite.config.ts define)
  // @ts-ignore
  if (typeof process !== 'undefined' && process.env && process.env.GROQ_API_KEY) {
    // @ts-ignore
    return process.env.GROQ_API_KEY;
  }
  
  return "";
};

const getOpenRouterKey = (): string => {
  // 1. Check Import Meta
  const meta = import.meta as any;
  if (meta.env && meta.env.VITE_OPENROUTER_API_KEY) return meta.env.VITE_OPENROUTER_API_KEY;
  
  // 2. Check Process Env
  // @ts-ignore
  if (typeof process !== 'undefined' && process.env && process.env.OPENROUTER_API_KEY) {
    // @ts-ignore
    return process.env.OPENROUTER_API_KEY;
  }

  return "";
};

const GROQ_API_KEY = getApiKey();

// --- CLIENT-SIDE COMPRESSION HELPER ---
// Replaces server-side sharp. Resizes to max 1024px and converts to JPEG 70% to avoid 413 Payload errors.
const compressImage = (base64Str: string): Promise<string> => {
  return new Promise((resolve) => {
    // Safety timeout to prevent hanging if image fails to load
    const timeoutId = setTimeout(() => resolve(base64Str), 4000);

    const img = new Image();
    img.src = base64Str;
    img.crossOrigin = "Anonymous";
    
    img.onload = () => {
      clearTimeout(timeoutId);
      const maxWidth = 1024;
      let width = img.width;
      let height = img.height;

      // Resize logic: Maintain aspect ratio
      if (width > maxWidth) {
        height = Math.round((height * maxWidth) / width);
        width = maxWidth;
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        resolve(base64Str); // Fallback if canvas not supported
        return;
      }

      // Draw white background (for PNG transparency to JPEG)
      ctx.fillStyle = "#FFFFFF";
      ctx.fillRect(0, 0, width, height);
      
      // Draw image
      ctx.drawImage(img, 0, 0, width, height);
      
      // Export as JPEG at 70% quality
      const compressed = canvas.toDataURL('image/jpeg', 0.7);
      resolve(compressed);
    };

    img.onerror = (err) => {
      clearTimeout(timeoutId);
      console.warn("Compression failed, sending original:", err);
      resolve(base64Str);
    };
  });
};

// --- Helper: Call Groq (Text/Logic) ---
async function callGroq(
  messages: any[], 
  model = MODEL_BRAIN, 
  jsonMode = true,
  temperature = 0.6
) {
  if (!GROQ_API_KEY) {
    throw new Error("Groq API Key is missing. Please add VITE_GROQ_API_KEY to your Vercel Environment Variables.");
  }

  try {
    const response = await fetch(GROQ_API_URL, {
      method: "POST",
      headers: {
        "Authorization": `Bearer ${GROQ_API_KEY}`,
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        messages: messages,
        model: model,
        temperature: temperature,
        response_format: jsonMode ? { type: "json_object" } : undefined,
        max_tokens: 4096
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`Groq API Error: ${errText}`);
    }

    const data = await response.json();
    
    if (!data.choices || !data.choices.length) {
      if (data.error) throw new Error(`Groq API Error: ${data.error.message || JSON.stringify(data.error)}`);
      throw new Error(`Groq API returned unexpected format (No choices): ${JSON.stringify(data)}`);
    }

    return data.choices[0].message.content;
  } catch (e: any) {
    console.error("Groq Call Failed:", e);
    throw e;
  }
}

// --- Helper: Call Vision (Groq or OpenRouter) ---
async function callVisionAPI(
  messages: any[],
  provider: 'GROQ' | 'OPENROUTER',
  apiKey: string,
  overrideModel?: string // Allow passing a specific model for retries
) {
  const url = provider === 'GROQ' ? GROQ_API_URL : OPENROUTER_API_URL;
  
  // Determine model
  let model = provider === 'GROQ' ? MODEL_VISION_GROQ : MODEL_VISION_FREE_PRIMARY;
  if (overrideModel) {
    model = overrideModel;
  }
  
  // Ensure clean key
  const cleanKey = apiKey?.trim();

  if (!cleanKey) {
     throw new Error(`API Key is missing for ${provider}. Please check your Vercel Environment Variables.`);
  }

  const headers: any = {
    "Authorization": `Bearer ${cleanKey}`,
    "Content-Type": "application/json"
  };

  // OpenRouter specific headers to avoid 403/500 on some routes
  if (provider === 'OPENROUTER') {
    headers["HTTP-Referer"] = "https://tubemaster.ai";
    headers["X-Title"] = "TubeMaster AI";
  }

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: headers,
      body: JSON.stringify({
        model: model,
        messages: messages,
        // Llama Vision on Groq supports JSON mode but sometimes it's safer to prompt for it
        // We keep it on for now as Llama 3.2 supports it
        response_format: { type: "json_object" },
        max_tokens: 2048
      })
    });

    if (!response.ok) {
      const errText = await response.text();
      throw new Error(`${provider} API Error: ${errText}`);
    }

    const data = await response.json();
    
    if (!data.choices || !data.choices.length) {
      if (data.error) throw new Error(`${provider} API Error: ${data.error.message || JSON.stringify(data.error)}`);
      throw new Error(`${provider} API returned unexpected format (No choices): ${JSON.stringify(data)}`);
    }

    return data.choices[0].message.content;
  } catch (e: any) {
    console.error(`${provider} Vision Failed:`, e);
    throw e;
  }
}

/**
 * Robust JSON cleaner. 
 */
const cleanJson = (text: string): string => {
  if (!text) return "{}";
  let clean = text.replace(/```json\s*/g, '').replace(/```\s*$/g, '');
  const firstBrace = clean.indexOf('{');
  const lastBrace = clean.lastIndexOf('}');
  if (firstBrace !== -1 && lastBrace !== -1 && lastBrace > firstBrace) {
    clean = clean.substring(firstBrace, lastBrace + 1);
  }
  return clean;
};

// --- Keyword Finder ---
export const findKeywords = async (topic: string): Promise<any[]> => {
  const prompt = `Act as a world-class YouTube SEO Expert.
    Analyze the topic: "${topic}" and generate exactly 10 high-potential keywords.
    For EACH keyword, apply these 10 Logic Points:
    1. Search Volume: Estimate monthly searches.
    2. Difficulty (KD): 0-100 score.
    3. Opportunity Score: 0-100 score.
    4. Trend: Rising, Stable, Falling, Seasonal.
    5. Intent: Informational, Educational, Entertainment, Commercial.
    6. CPC: Estimate value ($).
    7. Competition Density: Low, Medium, High.
    8. Top Competitor: Name a likely channel.
    9. Video Age Avg: Fresh or Old.
    10. CTR Potential: High/Avg/Low.
    
    Return strictly a JSON Object with a key "keywords" containing an array of objects.`;

  const jsonStr = await callGroq([{ role: "user", content: prompt }], MODEL_BRAIN, true);
  const parsed = JSON.parse(cleanJson(jsonStr));
  return parsed.keywords || parsed; 
};

// --- Competitor Analysis ---
export const analyzeCompetitor = async (channelUrl: string): Promise<any> => {
  let pageText = "";
  try {
    const proxyUrl = `https://api.allorigins.win/get?url=${encodeURIComponent(channelUrl)}`;
    const response = await fetch(proxyUrl);
    if (!response.ok) throw new Error("Proxy Error");
    const data = await response.json();
    if (!data.contents) throw new Error("Could not fetch channel page");
    
    const html = data.contents;
    const titleMatch = html.match(/<title>(.*?)<\/title>/);
    const descMatch = html.match(/name="description" content="(.*?)"/);
    
    pageText = `Channel: ${titleMatch?.[1]}\nDesc: ${descMatch?.[1]}`;
  } catch (e) {
    pageText = `Channel URL: ${channelUrl} (Scrape failed)`;
  }

  const prompt = `Analyze this YouTube channel: ${pageText.substring(0, 2000)} 
  Identify 3 strengths, 3 weaknesses, 3 content gaps, and an action plan.
  Return strictly JSON.`;

  const jsonStr = await callGroq([{ role: "user", content: prompt }], MODEL_BRAIN, true);
  return JSON.parse(cleanJson(jsonStr));
};

// --- Script Generator ---
export const generateScript = async (title: string, audience: string): Promise<any> => {
  const prompt = `Write a YouTube script for "${title}" targeting "${audience}".
    Logic sections: Hook, Stakes, Context, Twist, Value, Retention Spike, Emotion, Re-engagement, Payoff.
    Return strictly JSON with keys: title, estimatedDuration, targetAudience, sections (array).`;

  const jsonStr = await callGroq([{ role: "user", content: prompt }], MODEL_BRAIN, true);
  return JSON.parse(cleanJson(jsonStr));
};

// --- Thumbnail Generator ---
export const generateThumbnail = async (prompt: string, style: string, mood: string, optimize: boolean): Promise<ThumbnailGenResult> => {
  let finalPrompt = prompt;
  let optimizedPrompt = prompt;

  if (optimize) {
    try {
      const optRes = await callGroq(
        [{ role: "user", content: `Rewrite this image prompt for high-CTR. Style: ${style}. Mood: ${mood}. Original: "${prompt}". Output ONLY prompt.` }],
        MODEL_BRAIN,
        false 
      );
      optimizedPrompt = optRes;
      finalPrompt = optimizedPrompt;
    } catch (e) {
      console.warn("Optimization failed");
    }
  }

  const seed = Math.floor(Math.random() * 1000000);
  const imageUrl = `https://image.pollinations.ai/prompt/${encodeURIComponent(finalPrompt)}?width=1280&height=720&seed=${seed}&model=flux&nologo=true`;

  return {
    imageUrl: imageUrl,
    originalPrompt: prompt,
    optimizedPrompt: optimizedPrompt,
    style: style,
    createdAt: Date.now()
  };
};

// --- Title Generator ---
export const generateTitles = async (topic: string): Promise<string[]> => {
  const jsonStr = await callGroq(
    [{ role: "user", content: `Generate 10 click-worthy titles for: "${topic}". JSON object with key "titles".` }],
    MODEL_BRAIN,
    true
  );
  const parsed = JSON.parse(cleanJson(jsonStr));
  return parsed.titles || parsed;
};

// --- Best Time ---
export const suggestBestTime = async (title: string, audience: string, tags: string = ""): Promise<string> => {
  return await callGroq(
    [{ role: "user", content: `Best time to publish: "${title}" for "${audience}". Explain why.` }],
    MODEL_BRAIN,
    false
  );
};

// --- Thumbnail Compare (Vision Mode) ---
export const compareThumbnailsText = async (descA: string, descB: string): Promise<any> => {
  // Kept for type compatibility, but effectively deprecated by UI
  return { winner: 'A', scoreA: 0, scoreB: 0, reasoning: "Text mode deprecated.", breakdown: [] };
};

export const compareThumbnailsVision = async (
  imgABase64: string, 
  imgBBase64: string, 
  provider: 'GROQ' | 'OPENROUTER',
  userKey?: string
): Promise<any> => {
  
  // IMPORTANT: If userKey is provided by UI input, use it. 
  // Otherwise fallback to environment variable.
  const apiKey = (userKey && userKey.trim().length > 0) 
    ? userKey 
    : (provider === 'GROQ' ? GROQ_API_KEY : getOpenRouterKey());
  
  if (!apiKey) {
    throw new Error(`API Key missing for ${provider} provider. Please enter your key in the settings or configured Vercel Env Variables.`);
  }

  // 1. COMPRESSION STEP: Compress images before sending to API
  // Resizes to max 1024px and JPEG 70% to prevent 413 errors
  const [compressedA, compressedB] = await Promise.all([
    compressImage(imgABase64),
    compressImage(imgBBase64)
  ]);

  // 2. ANTI-BIAS MECHANISM: Randomly swap images to prevent "Second Image Bias"
  // LLMs often favor the last image they see. By randomizing, we mitigate this.
  const isSwapped = Math.random() > 0.5;
  const image1 = isSwapped ? compressedB : compressedA; // "Image 1" in prompt
  const image2 = isSwapped ? compressedA : compressedB; // "Image 2" in prompt

  // 3. RIGOROUS SYSTEM PROMPT: Enforces 10-point logic & Subject Detection
  const analysisPrompt = `
    Act as a specialized YouTube Thumbnail Optimization AI.
    
    You are comparing two thumbnails (**Image 1** and **Image 2**) for a high-stakes A/B test.
    
    **CONTEXT**: 
    - 70% of YouTube views are on Mobile. Small screens.
    - High CTR requires immediate comprehension.
    - "B-Roll" or "Artistic" wide shots usually FAIL.
    - "Emotive Close-ups" usually WIN.

    **YOUR TASK**:
    Evaluate the images based on these 10 PROVEN VIRAL FACTORS:

    1. **Mobile Clarity (The Squint Test)**: If you squint, can you still see the main subject? (Weight: 20%)
    2. **Facial Dominance**: Is the face large and emotive? (Close-up > Full Body). (Weight: 15%)
    3. **Text Readability**: Large, bold, contrasting text? (<5 words). (Weight: 10%)
    4. **Curiosity Gap**: Does the image provoke a "Must Click" question? (Weight: 10%)
    5. **Color Vibrancy**: High saturation/contrast vs dull/washed out. (Weight: 10%)
    6. **Subject Isolation**: distinct separation from background. (Weight: 10%)
    7. **Rule of Thirds**: Professional composition. (Weight: 5%)
    8. **Emotional Impact**: Extreme emotion (Shock/Fear/Joy) > Neutral. (Weight: 10%)
    9. **Visual Hierarchy**: Clear focal point. (Weight: 5%)
    10. **Lighting Quality**: Professional vs Amateur/Dark. (Weight: 5%)

    **CRITICAL BIAS CORRECTION**:
    - You may have a bias towards the second image. IGNORE IT.
    - You may have a bias towards "pretty" art. IGNORE IT. 
    - **PRIORITIZE CLOSE-UPS.** If Image 1 is a face close-up and Image 2 is a wide landscape, Image 1 wins 9 times out of 10.

    **OUTPUT FORMAT (JSON ONLY)**:
    {
      "shot_type_1": "Identify shot type of Image 1 (Close-up/Mid/Wide)",
      "shot_type_2": "Identify shot type of Image 2 (Close-up/Mid/Wide)",
      "winner": "1" or "2",
      "score1": (0-10 float), 
      "score2": (0-10 float),
      "reasoning": "Direct explanation of why the winner gets more clicks on mobile.",
      "breakdown": [
        { "criterion": "Mobile Clarity", "winner": "1" or "2", "explanation": "..." },
        ... (for all 10 factors)
      ]
    }
  `;

  const messages = [
    {
      role: "user",
      content: [
        { type: "text", text: analysisPrompt },
        { type: "image_url", image_url: { url: image1 } }, // First Image (1)
        { type: "image_url", image_url: { url: image2 } }  // Second Image (2)
      ]
    }
  ];

  try {
    // Try Primary Strategy
    const jsonStr = await callVisionAPI(messages, provider, apiKey);
    const rawResult = JSON.parse(cleanJson(jsonStr));

    // MAP RESULTS BACK TO ORIGINAL A/B
    // If isSwapped is TRUE: Image 1 is User's B, Image 2 is User's A.
    
    const mapWinner = (w: string) => {
      const cleanW = w.toString().trim();
      if (cleanW === '1' || cleanW.toLowerCase().includes('image 1')) return isSwapped ? 'B' : 'A';
      if (cleanW === '2' || cleanW.toLowerCase().includes('image 2')) return isSwapped ? 'A' : 'B';
      return 'A'; // Fallback
    };

    // Handle potentially different JSON structures from different models (fallback safety)
    const score1 = rawResult.score1 || 0;
    const score2 = rawResult.score2 || 0;

    const finalResult = {
      winner: mapWinner(rawResult.winner),
      scoreA: isSwapped ? score2 : score1,
      scoreB: isSwapped ? score1 : score2,
      reasoning: rawResult.reasoning,
      breakdown: rawResult.breakdown ? rawResult.breakdown.map((item: any) => ({
        criterion: item.criterion,
        winner: mapWinner(item.winner),
        explanation: item.explanation
      })) : []
    };

    return finalResult;

  } catch (error: any) {
    if (provider === 'OPENROUTER') {
      console.warn("Primary OpenRouter model failed. Attempting fallback...");
      try {
        const fallbackJsonStr = await callVisionAPI(messages, provider, apiKey, MODEL_VISION_FREE_FALLBACK);
        const rawResultFallback = JSON.parse(cleanJson(fallbackJsonStr));
        
        // Same mapping logic for fallback
        const mapWinner = (w: string) => {
          const cleanW = w.toString().trim();
          if (cleanW === '1' || cleanW.toLowerCase().includes('image 1')) return isSwapped ? 'B' : 'A';
          if (cleanW === '2' || cleanW.toLowerCase().includes('image 2')) return isSwapped ? 'A' : 'B';
          return 'A';
        };

        const score1 = rawResultFallback.score1 || 0;
        const score2 = rawResultFallback.score2 || 0;

        return {
          winner: mapWinner(rawResultFallback.winner),
          scoreA: isSwapped ? score2 : score1,
          scoreB: isSwapped ? score1 : score2,
          reasoning: rawResultFallback.reasoning,
          breakdown: rawResultFallback.breakdown ? rawResultFallback.breakdown.map((item: any) => ({
            criterion: item.criterion,
            winner: mapWinner(item.winner),
            explanation: item.explanation
          })) : []
        };
      } catch (fallbackError: any) {
        throw new Error(`Primary & Fallback Models Failed. Error: ${error.message}`);
      }
    }
    throw error;
  }
};
