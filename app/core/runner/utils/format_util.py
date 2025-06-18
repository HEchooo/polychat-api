import re

def extract_http_links(text):
    basic_pattern = r'https?://[^\s]+'
    match = re.search(basic_pattern, text)
    
    if not match:
        return ""
    
    link = match.group()
    
    if any(domain in link.lower() for domain in ['tb.cn', 'taobao.com', 'tmall.com']):
        link_end = match.end()
        remaining_text = text[link_end:]
        
        space_param_match = re.match(r'\s+([A-Za-z0-9]+)', remaining_text)
        
        if space_param_match:
            return link + ' ' + space_param_match.group(1)
    
    return link