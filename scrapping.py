import requests
from bs4 import BeautifulSoup
import json
import concurrent.futures
import time
import re
from urllib.parse import urljoin, urlparse
import logging
import random
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

BASE_URL = 'https://www.shl.com'
CATALOG_URL = f'{BASE_URL}/solutions/products/product-catalog/'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
    'Cache-Control': 'max-age=0'
}
MAX_WORKERS = 3
RETRY_ATTEMPTS = 3
DELAY_BETWEEN_REQUESTS = 3

def collect_assessment_links():
    all_links = []
    seen_links = set()
    
    types_to_collect = [1, 2]
    
    for type_value in types_to_collect:
        start = 0
        items_per_page = 12
        max_items = 500
        consecutive_empty_pages = 0
        
        logging.info(f"Starting collection for type={type_value}")
        
        while start <= max_items:
            if start == 0:
                url = CATALOG_URL
                if type_value != 1:
                    url = f"{CATALOG_URL}?type={type_value}"
            elif start == items_per_page:
                url = f"{CATALOG_URL}?start={start}&type={type_value}"
            else:
                url = f"{CATALOG_URL}?start={start}&type={type_value}&type={type_value}"
            
            logging.info(f"Collecting links from page with start={start}, type={type_value}: {url}")
            
            try:
                response = requests.get(url, headers=HEADERS)
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                logging.error(f"Request failed for start={start}, type={type_value}: {e}")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            links = []
            assessment_links = soup.find_all('a', href=True)
            
            for link in assessment_links:
                href = link.get('href')
                if '/view/' in href:
                    full_url = urljoin(BASE_URL, href)
                    if full_url not in seen_links:
                        links.append(full_url)
                        seen_links.add(full_url)
            
            logging.info(f"Found {len(links)} new links at start={start}, type={type_value}")
            if len(links) == 0:
                consecutive_empty_pages += 1
                if consecutive_empty_pages >= 2:
                    logging.info(f"No new links found for {consecutive_empty_pages} consecutive pages for type={type_value}. Moving to next type.")
                    break
            else:
                consecutive_empty_pages = 0
                
            all_links.extend(links)
            start += items_per_page
            time.sleep(DELAY_BETWEEN_REQUESTS)
    
    logging.info(f"Total unique links collected across all types: {len(all_links)}")
    return all_links

def is_valid_assessment_page(html_content):
    patterns = [
        r'assessment',
        r'test (type|duration)',
        r'competency',
        r'aptitude',
        r'personality',
        r'product (details|information)',
        r'technical (ability|skill)',
        r'test (length|time)',
        r'evaluation',
        r'measurement'
    ]
    
    for pattern in patterns:
        if re.search(pattern, html_content, re.I):
            return True
    return False

def extract_duration_from_text(soup):
    duration_tag = soup.find('h4', string=lambda t: t and ('Assessment length' in t or 'Duration' in t or 'Time' in t))
    if duration_tag and duration_tag.find_next_sibling('p'):
        return duration_tag.find_next_sibling('p').text.strip()
    
    duration_patterns = [
        r'Assessment length:?\s*([^<>\n]+)',
        r'(?:Approximate\s+)?(?:Completion|Testing)\s+Time:?\s*(\d+(?:-\d+)?\s*minutes)',
        r'(?:Duration|Test Length):?\s*(\d+(?:-\d+)?\s*minutes)',
        r'(\d+(?:-\d+)?)\s*minutes to complete',
        r'takes\s+(\d+(?:-\d+)?)\s*minutes',
        r'time:?\s*(\d+(?:-\d+)?(?:\s*to\s*\d+)?\s*minutes)',
        r'approximately\s+(\d+(?:-\d+)?)\s*minutes',
        r'test\s+duration:?\s*(\d+(?:-\d+)?)\s*minutes',
        r'(\d+(?:\s*-\s*\d+)?)\s*min(?:utes)?'
    ]
    
    for pattern in ['duration', 'time', 'length']:
        elements = soup.find_all(string=re.compile(pattern, re.I))
        for element in elements:
            parent = element.parent
            if parent:
                next_sibling = parent.find_next_sibling()
                if next_sibling and next_sibling.name in ['p', 'div', 'span']:
                    text = next_sibling.text.strip()
                    for duration_pattern in duration_patterns:
                        match = re.search(duration_pattern, text, re.I)
                        if match:
                            return match.group(1).strip()
    
    duration_texts = deep_text_search(soup, [
        r'time', r'duration', r'minutes', r'min', r'complete', r'length'
    ])
    
    for text in duration_texts:
        for pattern in duration_patterns:
            match = re.search(pattern, text, re.I)
            if match:
                return match.group(1).strip()
    
    return "N/A"

def deep_text_search(soup, key_patterns):
    matches = []
    for text in soup.stripped_strings:
        text_str = text.strip()
        if text_str:
            for pattern in key_patterns:
                if re.search(pattern, text_str, re.I):
                    matches.append(text_str)
                    break
    return matches

def extract_assessment_details(url):
    for attempt in range(RETRY_ATTEMPTS):
        try:
            logging.info(f"Fetching details from {url}")
            response = requests.get(url, headers=HEADERS, timeout=15)
            response.raise_for_status()
            html_content = response.text
            if not is_valid_assessment_page(html_content):
                logging.info(f"Skipping {url} - not a valid assessment page")
                return None
                
            soup = BeautifulSoup(html_content, 'html.parser')
            category_mapping = {
                'A': 'Ability & Aptitude',
                'B': 'Biodata & Situational Judgement',
                'C': 'Competencies',
                'D': 'Development & 360',
                'E': 'Assessment Exercises',
                'K': 'Knowledge & Skills',
                'P': 'Personality & Behavior',
                'S': 'Simulations'
            }
            
            assessment = {
                'name': 'Unknown',
                'url': url,
                'description': 'N/A',
                'duration': 'N/A',
                'test_type': 'N/A',
                'remote_testing': 'No',
                'adaptive_irt': 'No'
            }
            name_selectors = [
                'h1', '.product-title', '.assessment-title', '.page-title', 'header h1',
                '.hero-title', '.entry-title', '.title'
            ]
            for selector in name_selectors:
                name_elem = soup.select_one(selector)
                if name_elem and name_elem.text.strip():
                    assessment['name'] = name_elem.text.strip()
                    break
            desc_tag = soup.find('h4', string='Description')
            if desc_tag and desc_tag.find_next_sibling('p'):
                assessment['description'] = desc_tag.find_next_sibling('p').text.strip()
            else:
                desc_selectors = [
                    '.description', '.product-description', '.assessment-description',
                    '.content', '.product-content', '.overview', '.product-overview',
                    '.entry-content p', 'article p', '.product-details p', 
                    '.tab-content p', '.product-info p', '.panel-body p'
                ]
                
                description_texts = []
                for selector in desc_selectors:
                    desc_elems = soup.select(selector)
                    if desc_elems:
                        for elem in desc_elems:
                            text = elem.text.strip()
                            if text and len(text) > 30:
                                description_texts.append(text)
                                
                if assessment['name'] != 'Unknown':
                    name_elem = None
                    for selector in name_selectors:
                        name_elem = soup.select_one(selector)
                        if name_elem and name_elem.text.strip() == assessment['name']:
                            break
                    
                    if name_elem:
                        parent = name_elem.parent
                        siblings = []
                        for _ in range(3):
                            if parent:
                                p_tags = parent.find_all('p')
                                for p in p_tags:
                                    text = p.text.strip()
                                    if text and len(text) > 30 and text not in description_texts:
                                        siblings.append(text)
                                parent = parent.parent
                        
                        if siblings:
                            description_texts.extend(siblings)
                meta_desc = soup.find('meta', attrs={'name': 'description'})
                if meta_desc and meta_desc.get('content'):
                    meta_content = meta_desc['content'].strip()
                    if meta_content and len(meta_content) > 30:
                        description_texts.append(meta_content)
                        
                if description_texts:
                    unique_texts = list(set([re.sub(r'\s+', ' ', text) for text in description_texts]))
                    unique_texts = [text for text in unique_texts if len(text) > 30]
                    
                    if unique_texts:
                        if len(unique_texts) > 1:
                            unique_texts.sort(key=len, reverse=True)
                            assessment['description'] = unique_texts[0]
                            if len(unique_texts) > 1 and len(unique_texts[1]) > 100:
                                assessment['description'] += " " + unique_texts[1]
                        else:
                            assessment['description'] = unique_texts[0]
            duration = extract_duration_from_text(soup)
            if duration != "N/A":
                assessment['duration'] = duration
                
            if assessment['duration'] == 'N/A':
                duration_patterns = [
                    r'(?:Approximate\s+)?(?:Completion|Testing)\s+Time:?\s*(\d+(?:-\d+)?\s*minutes)',
                    r'(?:Duration|Test Length):?\s*(\d+(?:-\d+)?\s*minutes)',
                    r'(\d+(?:-\d+)?)\s*minutes to complete',
                    r'takes\s+(\d+(?:-\d+)?)\s*minutes',
                    r'time:?\s*(\d+(?:-\d+)?(?:\s*to\s*\d+)?\s*minutes)',
                    r'approximately\s+(\d+(?:-\d+)?)\s*minutes',
                    r'test\s+duration:?\s*(\d+(?:-\d+)?)\s*minutes',
                    r'(\d+(?:\s*-\s*\d+)?)\s*min(?:utes)?'
                ]
                
                for pattern in duration_patterns:
                    match = re.search(pattern, html_content, re.I)
                    if match:
                        assessment['duration'] = match.group(1).strip()
                        break
            test_type_label = soup.find(string=re.compile(r'Test\s+Type:', re.I))
            if test_type_label:
                parent_element = test_type_label.parent
                test_type_text = parent_element.get_text() if parent_element else ""
                
                if not re.search(r'[ABCDEKPS]', test_type_text, re.I) and parent_element:
                    next_element = parent_element.find_next_sibling()
                    if next_element:
                        test_type_text += " " + next_element.get_text()
                
                type_codes = re.findall(r'[ABCDEKPS]', test_type_text, re.I)
                if type_codes:
                    types = [category_mapping.get(code.upper(), code) for code in type_codes]
                    assessment['test_type'] = ', '.join(types)
            if assessment['test_type'] == 'N/A':
                url_match = re.search(r'/([ABCDEKPS])/view/', url, re.I)
                if url_match:
                    category_code = url_match.group(1).upper()
                    if category_code in category_mapping:
                        assessment['test_type'] = category_mapping[category_code]
                if assessment['test_type'] == 'N/A':
                    breadcrumb = soup.find('nav', class_='breadcrumb') or soup.find('ol', class_='breadcrumb')
                    if breadcrumb:
                        for category_code, category_name in category_mapping.items():
                            if re.search(re.escape(category_name), breadcrumb.text, re.I):
                                assessment['test_type'] = category_name
                                break
                if assessment['test_type'] == 'N/A':
                    for category_code, category_name in category_mapping.items():
                        category_pattern = rf'\b{category_code}[\s\-:]+{re.escape(category_name)}\b'
                        if re.search(category_pattern, html_content, re.I):
                            assessment['test_type'] = category_name
                            break
                        if re.search(rf'\b{re.escape(category_name)}\b', html_content, re.I):
                            assessment['test_type'] = category_name
                            break
                if assessment['test_type'] == 'N/A':
                    type_keywords = {
                        'Ability & Aptitude': ['aptitude', 'ability', 'cognitive', 'reasoning', 'numerical', 'verbal', 'abstract', 'logical', 'inductive'],
                        'Biodata & Situational Judgement': ['situational', 'judgement', 'judgment', 'sjt', 'scenario', 'biodata'],
                        'Competencies': ['competenc', 'skill assessment', 'proficiency', 'capability'],
                        'Development & 360': ['development', '360', 'feedback', 'review', 'evaluation'],
                        'Assessment Exercises': ['exercise', 'simulation', 'role play', 'in-basket', 'inbox'],
                        'Knowledge & Skills': ['knowledge', 'technical', 'skill test', 'proficiency test'],
                        'Personality & Behavior': ['personality', 'behavior', 'behavioural', 'behavioral', 'trait', 'preference'],
                        'Simulations': ['simulation', 'realistic', 'scenario', 'interactive', 'v']
                    }
                    
                    for test_type, keywords in type_keywords.items():
                        for keyword in keywords:
                            if re.search(r'\b' + keyword + r'\b', html_content, re.I):
                                assessment['test_type'] = test_type
                                break
                        if assessment['test_type'] != 'N/A':
                            break
            assessment['remote_testing'] = 'Yes' if re.search(r'\bremote(?:ly)?\s+test(?:ing)?\b', html_content, re.I) else 'No'
            assessment['adaptive_irt'] = 'Yes' if re.search(r'\badaptive\s+IRT\b', html_content, re.I) else 'No'
            
            return assessment
        
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {url} (attempt {attempt + 1}/{RETRY_ATTEMPTS}): {e}")
            if attempt < RETRY_ATTEMPTS - 1:
                time.sleep(random.uniform(2, 5))
            else:
                logging.error(f"Failed to extract details from {url} after {RETRY_ATTEMPTS} attempts.")
                return {
                    'name': 'Failed to extract',
                    'url': url,
                    'description': 'N/A',
                    'duration': 'N/A',
                    'test_type': 'N/A',
                    'remote_testing': 'N/A',
                    'adaptive_irt': 'N/A'
                }
        except Exception as e:
            logging.exception(f"An unexpected error occurred while processing {url}: {e}")
            return {
                'name': 'Error during extraction',
                'url': url,
                'description': 'N/A',
                'duration': 'N/A',
                'test_type': 'N/A',
                'remote_testing': 'N/A',
                'adaptive_irt': 'N/A'
            }

def main():
    start_time = time.time()
    assessment_links = collect_assessment_links()
    
    if not assessment_links:
        logging.warning("No assessment links found. Exiting.")
        return
    
    all_assessments = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(extract_assessment_details, url) for url in assessment_links]
        for future in concurrent.futures.as_completed(futures):
            try:
                assessment = future.result()
                if assessment:
                    all_assessments.append(assessment)
            except Exception as e:
                logging.error(f"Exception during extraction: {e}")
    unique_assessments = []
    seen_urls = set()
    for assessment in all_assessments:
        if assessment['url'] not in seen_urls:
            unique_assessments.append(assessment)
            seen_urls.add(assessment['url'])
    unique_assessments.sort(key=lambda x: x['name'])
    
    output_filename = 'assessments.json'
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(unique_assessments, f, indent=4, ensure_ascii=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    logging.info(f"Total assessments extracted: {len(unique_assessments)}")
    logging.info(f"Extraction completed in {elapsed_time:.2f} seconds. Data saved to {output_filename}")

if __name__ == "__main__":
    main()