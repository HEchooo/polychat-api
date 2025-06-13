import re

def extract_http_links(text):
    pattern = r'https?://[^\s<>"{}|\\^`[\]]+'
    
    links = re.findall(pattern, text)
    
    cleaned_links = []
    for link in links:
        link = re.sub(r'[.,;:!?)\]}]+$', '', link)
        if link:
            cleaned_links.append(link)
    
    unique_links = []
    for link in cleaned_links:
        if link not in unique_links:
            unique_links.append(link)
    
    return unique_links
