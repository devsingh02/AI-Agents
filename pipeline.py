import easyocr
from PIL import Image
import numpy as np
import torch
import random
from transformers import AutoTokenizer, AutoModel, AutoProcessor, AutoModelForZeroShotImageClassification
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import os
import re
import nltk
from nltk.corpus import words
nltk.download('words')
nltk.download('punkt')
from nltk.metrics.distance import edit_distance
from difflib import SequenceMatcher
import cv2
import torch.nn.functional as F
import emoji
from tabulate import tabulate
import gc  # For garbage collection on CPU

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If you have GPU
device = torch.device("cpu") # If you have CPU

# MODEL 1: InternVL
try:
    internvl_model_path = os.path.join("Models", "InternVL2_5-1B-MPO")  # Path to InternVL model files

    # Load the model locally
    model_int = AutoModel.from_pretrained(
        internvl_model_path,
        torch_dtype=torch.bfloat16,
        # device_map=device,
        low_cpu_mem_usage=True,
        # use_flash_attn=True,
        trust_remote_code=True
    ).eval()

    # Disable dropout for inference
    for module in model_int.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0

    # Load the tokenizer locally
    tokenizer_int = AutoTokenizer.from_pretrained(internvl_model_path, trust_remote_code=True)

    # Verification for InternVL
    print("\nInternVL model and tokenizer loaded successfully.")
    print(f"InternVL model device: {next(model_int.parameters()).device}")
except Exception as e:
    print(f"\nError loading InternVL model or tokenizer: {e}")


# MODEL 2: EasyOCR
try:
    # MODEL 2: EasyOCR
    reader = easyocr.Reader(['en', 'hi'], gpu=False)  # Initialize EasyOCR Reader  # Initialize EasyOCR Reader
    print("\nEasyOCR reader initialized successfully.")

    # Verification for EasyOCR
    if reader:
        print("EasyOCR model is ready for inference.")
except Exception as e:
    print(f"\nError initializing EasyOCR reader: {e}")


# MODEL 3: CLIP (Zero-Shot Classifier)
try:
    clip_model_path = os.path.join("Models", "clip-vit-base-patch32")  # Path to CLIP model files

    # Load the processor and model locally
    processor_clip = AutoProcessor.from_pretrained(clip_model_path)
    model_clip = AutoModelForZeroShotImageClassification.from_pretrained(clip_model_path).to(device)

    # Verification for CLIP
    print("\nCLIP model and processor loaded successfully.")
    print(f"CLIP model device: {next(model_clip.parameters()).device}\n")
except Exception as e:
    print(f"\nError loading CLIP model or processor: {e}\n")


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Global seed initialization
set_seed(42)

def build_transform(input_size=448):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB')),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=mean, std=std)
    ])
transform = build_transform()

def easyocr_ocr(image):
    image_np = np.array(image)  # Convert to NumPy array
    results = reader.readtext(image_np, detail=1)  # Extract text with bounding box details

    # Clear resources
    del image_np
    gc.collect()  # Explicitly call garbage collector to free memory
    # torch.cuda.empty_cache() # If using cuda / GPU

    if not results:
        return ""
    # Sort results by (top-to-bottom, left-to-right) reading order
    sorted_results = sorted(results, key=lambda x: (x[0][0][1], x[0][0][0]))
    # Extract the text in sorted order
    ordered_text = " ".join([res[1] for res in sorted_results]).strip()
    return ordered_text

def intern(image, prompt, max_tokens):
    pixel_values = transform(image).unsqueeze(0).to(device).to(torch.bfloat16)
    with torch.no_grad():
        response, _ = model_int.chat(
            tokenizer_int,
            pixel_values,
            prompt,
            generation_config={
                "max_new_tokens": max_tokens,
                "do_sample": False,
                "num_beams": 1,
                "temperature": 1.0,
                "top_p": 1.0,
                "repetition_penalty": 1.0,
                "length_penalty": 1.0,
                "pad_token_id": tokenizer_int.pad_token_id  # Explicitly set the pad_token_id
            },
            history=None,
            return_history=True
        )
    # Clean up memory after processing
    del pixel_values  # Explicitly delete pixel_values to free memory
    gc.collect()
    return response


def clip(image, labels):
    # Process the image and labels
    processed = processor_clip(
        text=labels,
        images=image,
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Clean up memory after processing
    del image, labels  # Explicitly delete the image and labels
    gc.collect()

    return processed

def get_roi(image_path, *roi):
    image = Image.open(image_path).convert('RGB')
    width, height = image.size

    roi_x_start = int(width * roi[0])
    roi_y_start = int(height * roi[1])
    roi_x_end = int(width * roi[2])
    roi_y_end = int(height * roi[3])

    cropped_image = image.crop((roi_x_start, roi_y_start, roi_x_end, roi_y_end))
    return cropped_image

# Constants for ROIs
BODY = (0.0, 0.0, 1.0, 1.0)
TAG = (0.05, 0.62, 1.0, 0.65)
DTAG = (0.05, 0.592, 1.0, 0.622)
TNC = (0.02, 0.98, 1.0, 1.0)
CTA = (0.68, 0.655, 0.87, 0.675)
GNC = (0.5, 0.652, 0.93, 0.77)

### TAGLINES ###

# ptag = "Extract all the text from the image that is clear, sharp, and legible. Ignore text that is blurry, faint, distorted, or too small to read. Provide only the clean and clearly visible parts of the text."
ptag = "Extract all the text from the image accurately."
pemo = "Carefully analyze the image to detect emojis. Emojis are graphical icons (e.g., 😀, 🎉, ❤️) and not regular text, symbols, or characters. Examine the image step by step to ensure only graphical emojis are counted. If no emojis are found, respond with 'NUMBER OF EMOJIS: 0'. If emojis are present, count them and provide reasoning before giving the final answer in the format 'NUMBER OF EMOJIS: [count]'. Do not count text or punctuation as emojis."

def clean_text(text):
    return re.sub(r'[^a-zA-Z0-9]', '', text).strip().lower()

def are_strings_similar(str1, str2, max_distance=3, max_length_diff=2): # EasyOCR giving better results that GOT
    if str1 == str2:
        return True
    if abs(len(str1) - len(str2)) > max_length_diff:
        return False
    edit_distance_value = edit_distance(str1, str2)
    # Remove percentage-based adjustment
    return edit_distance_value <= max_distance

def blur_image(image, strength):
    image_np = np.array(image)
    blur_strength = int(strength * 50)  # Scale blur strength (higher values = more blur)
    blur_strength = max(1, blur_strength | 1)  # Ensure odd blur kernel size for GaussianBlur
    blurred_image = cv2.GaussianBlur(image_np, (blur_strength, blur_strength), 0)
    blurred_pil_image = Image.fromarray(blurred_image)
    return blurred_pil_image

def is_blank(text, limit=15):
    """Check if ROI is blank"""
    return len(text) < limit

def is_unreadable_tagline(htag, tag):
    """Check if the tagline is unreadable."""
    clean_htag = clean_text(htag)
    clean_tag = clean_text(tag)
    return not are_strings_similar(clean_htag, clean_tag)

def is_hyperlink_tagline(tag):
    """Check if the tagline contains a hyperlink."""
    substrings = ['www', '.com', 'http']
    return any(sub in tag for sub in substrings)

def is_price_tagline(tag):
    """Check if the tagline contains a price tag."""
    # Keywords to exclude
    exclude_keywords = ["crore", "thousand", "million", "billion", "trillion"]
    # Regex pattern to exclude (e.g., ₹10 lac, $1 million, etc.)
    exclude_pattern = r'(₹\.?\s?\d+\s*(lac|lacs|lakh|lakhs|cr|k))|(\brs\.?\s?\d+\s*(lac|lacs|lakh|lakhs|cr|k))|(\$\.?\s?\d+\s*(lac|lacs|lakh|lakhs|cr|k))'
    # Regex pattern to detect valid prices (e.g., ₹100, $50, rs.500, र100)
    price_pattern = r'(₹\s?\d+)|(\brs\.?\s?\d+)|(\$\s?\d+)|(र\d+)'
    
    # Check for exclude keywords
    if any(keyword in tag for keyword in exclude_keywords):
        return False
    # Check if exclude pattern matches
    if re.search(exclude_pattern, tag):
        return False
    # If no exclusions found, check for valid price pattern
    return bool(re.search(price_pattern, tag))

def is_multiple_emoji(emoji):
    """Check if the input contains multiple emojis."""
    words = emoji.split()
    last_word = words[-1]
    return last_word not in ['0', '1']

def is_incomplete_tagline(tag, is_eng):
    """Check if the tag is incomplete."""
    # Remove emojis and clean whitespace
    tag = emoji.replace_emoji(tag, '')
    tag = tag.strip()
    # Check for ellipsis variations
    if tag.endswith(('...', '..')):
        return True
    if not is_eng and tag.endswith(('.')):
        return True

    return False

### BODY ###

def string_similarity(a, b):
    """
    Calculate similarity ratio between two strings.
    Returns float between 0 and 1, where 1 means identical strings.
    """
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def find_similar_substring(text, keyword, threshold=0.9):
    """
    Check if the keyword (as a whole phrase) exists in the text with a similarity >= threshold.
    """
    # Convert both text and keyword to lowercase for case-insensitive comparison
    text = text.lower()
    keyword = keyword.lower()

    # Check if the entire keyword exists as a substring in the text
    if keyword in text:
        return True

    # If the keyword is not found exactly, check for similar phrases using fuzzy matching
    # Split the text into overlapping phrases of the same length as the keyword
    keyword_length = len(keyword.split())
    words = text.split()

    for i in range(len(words) - keyword_length + 1):
        phrase = ' '.join(words[i:i + keyword_length])
        similarity = string_similarity(phrase, keyword)
        if similarity >= threshold:
            return True

    return False

def is_risky(body):
    """
    Detects whether an image promotes gambling, gambling websites/apps,
    high-risk trading, or high-risk investments.
    Uses fuzzy matching to account for OCR errors.
    """
    body = re.sub(r'[^a-zA-Z0-9\u0966-\u096F\s]', '', body)
    # Keywords and phrases related to risky activities
    risky_keywords = [
        # General gambling terms
        "casino", "poker", "jackpot", "blackjack",
        "sports betting", "online casino", "slot machine", "pokies",

        # Gambling website and app names (Global and India-promoted)
        "stake", "betano", "bet365", "888casino", "ladbrokes", "betfair",
        "unibet", "skybet", "coral", "betway", "sportingbet", "betvictor", "partycasino", "casinocom", "jackpot city",
        "playtech", "meccabingo", "fanDuel", "betmobile", "10bet", "10cric",
        "pokerstars" "fulltiltpoker", "wsop",

        # Gambling websites and apps promoted or popular in India
        "dream11", "dreamll", "my11circle", "cricbuzz", "fantasy cricket", "sportz exchange", "fun88",
        "funbb", "funbeecom", "funbee", "rummycircle", "pokertiger", "adda52", "khelplay",
        "paytm first games", "fanmojo", "betking", "1xbet", "parimatch", "rajapoker",

        # High-risk trading and investment terms
        "win cash", "high risk trading", "win lottery",
        "high risk investment", "investment scheme",
        "get rich quick", "trading signals", "financial markets", "day trading",
        "options trading", "forex signals"
    ]

    # Check each keyword using fuzzy matching
    for keyword in risky_keywords:
        if find_similar_substring(body, keyword):
            return True

    return False

def is_prom_illegal_activity(body):
    """
    Detects the presence of phrases followed by an illegal activity in an image using MiniCPM OCR, allowing for words between the phrase and activity.
    """
    illegal_activities = [
        "hack", "hacking", "cheating", "cheat", "drugs", "drug", "steal", "stealing",
        "phishing", "phish", "piracy", "pirate", "fraud", "smuggling", "smuggle",
        "counterfeiting", "blackmailing", "blackmail", "extortion", "scamming", "scam",
        "identity theft", "illegal trading", "money laundering", "poaching", "poach",
        "trafficking", "illegal arms", "explosives", "bomb", "bombing", "fake documents"
    ]

    phrases = [
        "how to", "learn", "steps to", "guide to", "ways to",
        "tutorial on", "methods for", "process of",
        "tricks for", "shortcuts to", "make"
    ]

    # Check for phrases and activities in sequence
    for phrase in phrases:
        for activity in illegal_activities:
            # Use regex to find phrase followed by activity with any number of words in between
            pattern = rf"{re.escape(phrase)}.*?{re.escape(activity)}"
            if re.search(pattern, body):
                return True

    return False

def is_competitor(body):
    competitor_brands = [
        "motorola", "oppo", "vivo", "htc", "sony", "nokia", "honor", "huawei", "asus", "lg",
        "oneplus", "apple", "micromax", "lenovo", "gionee", "infocus", "lava", "panasonic","intex",
        "blackberry", "xiaomi", "philips", "godrej", "whirlpool", "blue star", "voltas",
        "hitachi", "realme", "poco", "iqoo", "toshiba", "skyworth", "redmi", "nokia", "lava"
    ]
    for brand in competitor_brands:
        if re.search(r'\b' + re.escape(brand) + r'\b', body):
            return True  # Competitor brand detected

    return False  # No competitor brand detected

### THEME ###

def destroy_text_roi(image, *roi_params):
    image_np = np.array(image)

    h, w, _ = image_np.shape
    x1 = int(roi_params[0] * w)
    y1 = int(roi_params[1] * h)
    x2 = int(roi_params[2] * w)
    y2 = int(roi_params[3] * h)

    roi = image_np[y1:y2, x1:x2]

    blurred_roi = cv2.GaussianBlur(roi, (75, 75), 0)
    noise = np.random.randint(0, 50, (blurred_roi.shape[0], blurred_roi.shape[1], 3), dtype=np.uint8)
    noisy_blurred_roi = cv2.add(blurred_roi, noise)
    image_np[y1:y2, x1:x2] = noisy_blurred_roi
    return Image.fromarray(image_np)

def offensive(image):
    image = destroy_text_roi(image, *TAG)

    appr_labels = [
        "Inappropriate Content: Violence, Blood, political promotion, drugs, alcohol, cigarettes, smoking, cruelty, nudity, illegal activities",
        "Appropriate Content: Games, entertainment, Advertisement, Fashion, Sun-glasses, Food, Food Ad, Fast Food, Woman or Man Model, Television, natural scenery, abstract visuals, art, everyday objects, sports, news, general knowledge, medical symbols, and miscellaneous benign content"
    ]

    # Generate inputs for the CLIP model
    inputs_appr = clip(image, appr_labels)

    # Ensure that we are working with torch no_grad() context to save memory during inference
    with torch.no_grad():
        outputs_appr = model_clip(**inputs_appr)

    logits_per_image_appr = outputs_appr.logits_per_image
    probs_appr = F.softmax(logits_per_image_appr, dim=1)

    # Extract probabilities
    inappropriate_prob = probs_appr[0][0].item()
    appropriate_prob = probs_appr[0][1].item()

    # Clear any references to large variables to free memory
    del inputs_appr, outputs_appr, logits_per_image_appr, probs_appr
    gc.collect()

    if inappropriate_prob > appropriate_prob:
        return True
    return False


def religious(image):
    religious_labels = [
        "Digital art or sports or news or  miscellaneous activity or miscellaneous item or Person or religious places or diya or deepak or festival or nature or earth imagery or scenery or Medical Plus Sign or Violence or Military",
        "Hindu Deity / OM or AUM or Swastik symbol",
        "Jesus Christ / Christianity Cross"
    ]

    # Generate inputs for the CLIP model
    inputs_religious = clip(image, religious_labels)

    # Ensure that we are working with torch no_grad() context to save memory during inference
    with torch.no_grad():
        outputs_religious = model_clip(**inputs_religious)

    logits_per_image_religious = outputs_religious.logits_per_image
    probs_religious = F.softmax(logits_per_image_religious, dim=1)

    # Find the index with the highest probability
    highest_score_index = torch.argmax(probs_religious, dim=1).item()

    # Clear any references to large variables to free memory
    del inputs_religious, outputs_religious, logits_per_image_religious, probs_religious
    gc.collect()

    if highest_score_index != 0:
        return True, religious_labels[highest_score_index]  # Return the specific religious symbol detected
    return False, None

def image_quality(image_path):
    """
    Check if an image is low resolution or poor quality.
    """
    # Define stricter thresholds for low resolution
    MIN_WIDTH = 720       # Minimum width in pixels
    MIN_HEIGHT = 1600      # Minimum height in pixels
    MIN_PIXEL_COUNT = 1000000  # Minimum total pixel count (width x height)
    PIXEL_VARIANCE_THRESHOLD = 50  # Variance below this indicates low-quality images

    try:
        # Open the image
        image = Image.open(image_path)
        width, height = image.size  # Get width and height
        pixel_count = width * height  # Calculate total pixel count

        # Check resolution thresholds
        if width < MIN_WIDTH or height < MIN_HEIGHT or pixel_count < MIN_PIXEL_COUNT:
            return {"Bad Image Quality": 1}  # Low Quality: Low Resolution

        # Calculate pixel variance for quality check
        grayscale_image = image.convert("L")  # Convert to grayscale
        pixel_array = np.array(grayscale_image)
        variance = np.var(pixel_array)

        # Check pixel variance
        if variance < PIXEL_VARIANCE_THRESHOLD:
            return {"Bad Image Quality": 1}  # Low Quality: Low Pixel Variance

        return {"Bad Image Quality": 0}  # High Quality

    except Exception as e:
        print(f"Error processing image: {e}")
        return {"Bad Image Quality": 1}  # Default to low quality on error


ROIS = [
    # Top section divided into 3 parts
    (0.0, 0.612, 0.33, 0.626),  # Top left
    (0.33, 0.612, 0.66, 0.626),  # Top middle
    (0.66, 0.612, 1.0, 0.626),  # Top right

    # Bottom section divided into 3 parts
    (0.0, 0.678, 0.33, 0.686),  # Bottom left
    (0.33, 0.678, 0.66, 0.686),  # Bottom middle
    (0.66, 0.678, 1.0, 0.686),  # Bottom right

    # Extreme Right section
    (0.95, 0.63, 1, 0.678),

    # Middle Section (between Tag and Click)
    (0.029, 0.648, 0.35, 0.658),  # Middle left
    (0.35, 0.648, 0.657, 0.658)  # Middle right
]

# Detection parameters
DETECTION_PARAMS = {
    'clahe_clip_limit': 2.0,
    'clahe_grid_size': (8, 8),
    'gaussian_kernel': (5, 5),
    'gaussian_sigma': 0,
    'canny_low': 20,  
    'canny_high': 80,  
    'hough_threshold': 15,  
    'min_line_length': 10,
    'max_line_gap': 5,
    'edge_pixel_threshold': 0.01
}

def detect_straight_lines(roi_img):
    """Enhanced edge detection focusing on straight lines."""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(
        clipLimit=DETECTION_PARAMS['clahe_clip_limit'],
        tileGridSize=DETECTION_PARAMS['clahe_grid_size']
    )
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(
        enhanced,
        DETECTION_PARAMS['gaussian_kernel'],
        DETECTION_PARAMS['gaussian_sigma']
    )
    edges = cv2.Canny(
        blurred,
        DETECTION_PARAMS['canny_low'],
        DETECTION_PARAMS['canny_high']
    )
    line_mask = np.zeros_like(edges)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=DETECTION_PARAMS['hough_threshold'],
        minLineLength=DETECTION_PARAMS['min_line_length'],
        maxLineGap=DETECTION_PARAMS['max_line_gap']
    )
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, 2)
    return line_mask

def simple_edge_detection(roi_img):
    """Simple edge detection."""
    gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray, 50, 150)

def ribbon(image_path):
    """Detect the presence of a ribbon in an image."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = image.shape[:2]
    edge_present = []

    for i, roi in enumerate(ROIS):
        x1, y1, x2, y2 = [int(coord * (w if i % 2 == 0 else h)) for i, coord in enumerate(roi)]
        roi_img = image[y1:y2, x1:x2]

        if i < 6:  # Straight line detection for ROIs 0-5
            edges = detect_straight_lines(roi_img)
            edge_present.append(np.sum(edges) > edges.size * DETECTION_PARAMS['edge_pixel_threshold'])
        else:  # Original method for ROIs 6-8
            edges = simple_edge_detection(roi_img)
            edge_present.append(np.any(edges))

    result = all(edge_present[:6]) and not edge_present[6] and not edge_present[7] and not edge_present[8]
    return {"No Ribbon": 0 if result else 1}

def is_english(text):
    """
    Check if the text contains only allowed characters:
    - English letters (a-zA-Z)
    - Hindi numerals (०-९)
    - The Hindi character "र"
    - Common special characters (e.g., spaces, commas, periods, etc.)
    Returns True if the text contains only allowed characters, otherwise False.
    """
    # Define the regex pattern for allowed characters
    allowed_pattern = re.compile(
        r'^[a-zA-Z०-९\u0930\s\.,!?\-;:"\'()]*$'  # English letters, Hindi numerals, र, spaces, and common special characters
    )

    # Check if the entire text matches the allowed pattern
    return bool(allowed_pattern.match(text))

def tagline(image_path):
    """Classify the tagline based on its content."""
    results = {
        "Empty/Illegible/Black Tagline": 0,
        "Multiple Taglines": 0,
        "Incomplete Tagline": 0,
        "Hyperlink": 0,
        "Price Tag": 0,
        "Excessive Emojis": 0
    }

    # Get the tagline ROI
    image = get_roi(image_path, *TAG)
    himage = blur_image(image, 0.3)
    easytag = easyocr_ocr(image).lower().strip()
    unr = easyocr_ocr(himage).lower().strip()

    if is_blank(easytag) or is_blank(unr):
        results["Empty/Illegible/Black Tagline"] = 1
        return results

    is_eng = is_english(easytag)
    if not is_eng:
        # Skip illegibility check for non-eng text
        results["Empty/Illegible/Black Tagline"] = 0
        # Use easytag for all other checks
        tag = easytag
    else:
        # Proceed with the original logic for English text
        Tag = intern(image, ptag, 25).strip()
        tag = Tag.lower()

        htag = intern(himage, ptag, 25).lower().strip()
        # Set Tagline to 1 if either missing or illegible
        if is_unreadable_tagline(htag, tag):
            results["Empty/Illegible/Black Tagline"] = 1
            
    # Check for completeness
    results["Incomplete Tagline"] = 1 if is_incomplete_tagline(tag, is_eng) else 0

    # Check for hyperlink
    results["Hyperlink"] = 1 if is_hyperlink_tagline(tag) else 0

    # Check for price tag
    results["Price Tag"] = 1 if is_price_tagline(tag) else 0

    # Check for double tagline
    imagedt = get_roi(image_path, *DTAG)
    dtag = easyocr_ocr(imagedt).strip()
    results["Multiple Taglines"] = 0 if is_blank(dtag) else 1

    # Check for multiple emojis
    emoji = intern(image, pemo, 100)
    results["Excessive Emojis"] = 1 if is_multiple_emoji(emoji) else 0

    return results

def tooMuchText(image_path):
    DRIB = (0.04, 0.625, 1.0, 0.677)
    DUP = (0, 0, 1.0, 0.25)
    DBEL = (0, 0.85, 1.0, 1)
    image = Image.open(image_path).convert('RGB')
    image = destroy_text_roi(image, *DRIB)
    image = destroy_text_roi(image, *DUP)
    image = destroy_text_roi(image, *DBEL)
    bd = easyocr_ocr(image).lower().strip()
    return {"Too Much Text": 1 if len(bd) > 55 else 0}

def theme(image_path):
    """Check theme appropriateness."""
    results = {}
    image = Image.open(image_path).convert('RGB')

    # Check for offensive content
    results["Inappropriate Content"] = 1 if offensive(image) else 0

    # Check for religious content
    is_religious, religious_label = religious(image)
    results["Religious Content"] = f"1 [{religious_label}]" if is_religious else "0"

    return results

def body(image_path):
    """Check body content for various criteria."""
    results = {}

    image = Image.open(image_path).convert('RGB')
    bd = intern(image, ptag, 500).lower()
    ocr_substitutions = {'0': 'o', '1': 'l', '!': 'l', '@': 'a', '5': 's', '8': 'b'}

    for char, substitute in ocr_substitutions.items():
        bd = bd.replace(char, substitute)
    bd = ' '.join(bd.split())

    results["High Risk Content"] = 1 if is_risky(bd) else 0
    results["Illegal Content"] = 1 if is_prom_illegal_activity(bd) else 0
    results["Competitor References"] = 1 if is_competitor(bd) else 0

    return results

def is_valid_english(text):
    """Checks if all words in the given text are valid English words"""
    english_words = set(words.words())
    cleaned_words = ''.join(c.lower() if c.isalnum() else ' ' for c in text).split()
    # Check if all words are in the English dictionary
    return all(word.lower() in english_words for word in cleaned_words)

def cta(image_path):
    """Check CTA validity."""
    image = get_roi(image_path, *CTA)
    cta = intern(image, ptag, 5).strip()
    veng = is_valid_english(cta)
    eng = is_english(cta)

    if '.' in cta or '..' in cta or '...' in cta:  # CTA contains ellipsis
        return {"Bad CTA": 1}

    # Check if CTA contains any emoji
    if any(emoji.is_emoji(c) for c in cta):
        return {"Bad CTA": 1}

    # RULES FOR ENGLISH :-
    # # Check if CTA is only 1 word
    # if eng and veng and len(cta.split()) == 1:  # Split the CTA text by spaces and check word count
    #     return {"Bad CTA": 1}

    clean_cta = clean_text(cta)  # Clean the CTA text (lowercase, remove spaces)
    print(len(clean_cta))

    if eng and len(clean_cta) <= 2:  # CTA has 2 or fewer letters
        return {"Bad CTA": 1}

    if len(clean_cta) > 15:  # CTA has 2 or fewer letters
        return {"Bad CTA": 1}

    # If none of the above conditions are met, CTA is valid
    return {"Bad CTA": 0}

def tnc(image_path):
    """Check for terms and conditions."""
    image = get_roi(image_path, *TNC)
    tnc = easyocr_ocr(image)
    clean_tnc = clean_text(tnc)

    return {"Terms & Conditions": 0 if is_blank(clean_tnc) else 1}

def gnc(image_path):
    """Check for gestures/coach marks and display the image."""
    pgnc = "Is there a HAND POINTER/EMOJI or a LARGE ARROW or ARROW POINTER? Answer only 'yes' or 'no'."

    image = get_roi(image_path, *GNC)
    gnc = intern(image, pgnc, 900).lower()

    return {"Visual Gesture or Icon": 1 if 'yes' in gnc else 0}

# ### MULTIPLE IMAGES ###
# def classify(image_path):
#     """Perform complete classification with detailed results."""
#     # Components to check
#     components = [
#         image_quality,
#         ribbon,
#         tagline,
#         tooMuchText,
#         theme,
#         body,
#         cta,
#         tnc,
#         gnc
#     ]

#     # Collect all results
#     all_results = {}
#     for component in components:
#         results = component(image_path)
#         all_results.update(results)

#     # Calculate final classification
#     final_classification = 1 if any(result == 1 for result in all_results.values()) else 0

#     # Determine Pass or Fail
#     classification_result = "Fail" if final_classification == 1 else "Pass"

#     # Prepare the table data
#     table_data = []
#     labels = [
#         "Bad Image Quality", "No Ribbon", "Empty/Illegible/Black Tagline", "Multiple Taglines",
#         "Incomplete Tagline", "Hyperlink", "Price Tag", "Excessive Emojis", "Too Much Text",
#         "Inappropriate Content", "Religious Content", "High Risk Content",
#         "Illegal Content", "Competitor References", "Bad CTA", "Terms & Conditions",
#         "Visual Gesture or Icon"
#     ]

#     # Collect labels responsible for failure
#     failure_labels = [label for label in labels if all_results.get(label, 0) == 1]

#     for label in labels:
#         result = all_results.get(label, 0)  # Default to 0 if the result is not found
#         table_data.append([label, result])

#     # Format the results as a table
#     result_table = tabulate(table_data, headers=["LABEL", "RESULT"], tablefmt="fancy_grid")

#     # Return the final classification, result table, and failure labels (if any)
#     return classification_result, result_table, failure_labels



### DUMMY INTERFACE FOR TESTING ###

import time
def classify(image_path):
    """Perform complete classification with detailed results."""
    # Simulate processing time for the entire function
    # time.sleep(0)

    # Directly set all label results to 0
    all_results = {
        "Bad Image Quality": 1,
        "No Ribbon": 0,
        "Empty/Illegible/Black Tagline": 1,
        "Multiple Taglines": 0,
        "Incomplete Tagline": 0,
        "Hyperlink": 0,
        "Price Tag": 0,
        "Excessive Emojis": 0,
        "Too Much Text": 0,
        "Inappropriate Content": 0,
        "Religious Content": 0,
        "High Risk Content": 0,
        "Illegal Content": 0,
        "Competitor References": 0,
        "Bad CTA": 0,
        "Terms & Conditions": 0,
        "Visual Gesture or Icon": 0
    }

    # Calculate final classification
    final_classification = 1 if any(result == 1 for result in all_results.values()) else 0

    # Determine Pass or Fail
    classification_result = "Fail" if final_classification == 1 else "Pass"

    # Prepare the table data
    table_data = []
    labels = list(all_results.keys())

    # Collect labels responsible for failure
    failure_labels = [label for label in labels if all_results[label] == 1]

    for label in labels:
        result = all_results[label]
        table_data.append([label, result])

    # Format the results as a table
    result_table = tabulate(table_data, headers=["LABEL", "RESULT"], tablefmt="fancy_grid")

    # Return the final classification, result table, and failure labels (if any)
    return classification_result, result_table, failure_labels
