URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MARKDOWN_IMAGE_PATTERN = re.compile(r"!\[[^\]]*\]\([^)]*\)")
HTML_TAG_PATTERN = re.compile(r"</?(?:p|div|br|ul|ol|li|h[1-6]|blockquote|table|tr|td|th|a|img|span|strong|em|code)\b[^>]*>", re.I)

def clean_text(raw_text):
    if not isinstance(raw_text, str):
        return ""
    cleaned_text = html.unescape(raw_text)
    cleaned_text = MARKDOWN_IMAGE_PATTERN.sub(" <IMG> ", cleaned_text)
    cleaned_text = URL_PATTERN.sub(" <URL> ", cleaned_text)
    cleaned_text = HTML_TAG_PATTERN.sub(" ", cleaned_text)
    cleaned_text = unicodedata.normalize("NFKC", cleaned_text)
    return " ".join(cleaned_text.split())

for column_name in ["title", "body_text"]:
    df[column_name] = df[column_name].map(clean_text)

df["text_all"] = (df["title"] + " " + df["body_text"]).str.lower()

df[["title", "body_text", "text_all"]].head(5)
