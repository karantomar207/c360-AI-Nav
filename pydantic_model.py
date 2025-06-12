import json
import re
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ValidationError, validator

logger = logging.getLogger(__name__)

# Your existing Pydantic models (from previous artifact)
class Course(BaseModel):
    course_name: str
    page_title: Optional[str] = None
    page_description: Optional[str] = None
    fee_detail: Optional[str] = None
    learning_term_details: Optional[str] = None
    course_highlight: Optional[str] = None
    
    @validator('*', pre=True)
    def validate_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

class Certificate(BaseModel):
    page_title: str
    page_description: Optional[str] = None
    fee_detail: Optional[str] = None
    learning_term_details: Optional[str] = None
    job_details: Optional[str] = None
    
    @validator('*', pre=True)
    def validate_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

class Job(BaseModel):
    title: str
    description: Optional[str] = None
    salary: Optional[str] = ""
    requirements: Optional[str] = None
    benefits: Optional[str] = None
    
    @validator('*', pre=True)
    def validate_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

class Ebooks(BaseModel):
    title: str
    description: Optional[str] = None
    author: Optional[str] = None
    topics: Optional[str] = None
    level: Optional[str] = None
    pdf_upload: Optional[str] = None
    
    @validator('*', pre=True)
    def validate_strings(cls, v):
        if isinstance(v, str):
            return v.strip()
        return str(v) if v is not None else ""

def validate_enhanced_data(enhanced_content: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Validate enhanced content using Pydantic models
    """
    try:
        # Extract JSON object from the response
        json_match = re.search(r'\{.*\}', enhanced_content, re.DOTALL)
        if not json_match:
            logger.warning("No JSON object found in LLM response")
            return {"courses": [], "certificates": [], "jobs": [], "ebooks": []}
        
        try:
            enhanced_all_data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing enhanced JSON: {e}")
            return {"courses": [], "certificates": [], "jobs": [], "ebooks": []}
        
        if not isinstance(enhanced_all_data, dict):
            logger.warning("Enhanced data is not a dictionary")
            return {"courses": [], "certificates": [], "jobs": [], "ebooks": []}
        
        # Initialize result structure
        validated_results = {
            "courses": [],
            "certificates": [],
            "jobs": [],
            "ebooks": []
        }
        
        # Model mapping
        model_mapping = {
            "courses": Course,
            "certificates": Certificate,
            "jobs": Job,
            "ebooks": Ebooks
        }
        
        # Validate each dataset
        for dataset_name, items in enhanced_all_data.items():
            if dataset_name not in model_mapping:
                logger.warning(f"Unknown dataset type: {dataset_name}")
                continue
                
            if not isinstance(items, list):
                logger.warning(f"Dataset '{dataset_name}' is not a list")
                continue
            
            model_class = model_mapping[dataset_name]
            validated_items = []
            
            for i, item in enumerate(items):
                try:
                    # Validate item using Pydantic model
                    validated_item = model_class(**item)
                    validated_items.append(validated_item.dict())
                    
                except ValidationError as e:
                    logger.warning(f"Skipping invalid {dataset_name} item at index {i}: {e}")
                    # Log specific validation errors for debugging
                    for error in e.errors():
                        logger.debug(f"Validation error in {dataset_name}[{i}]: {error}")
                    continue
                    
                except Exception as e:
                    logger.error(f"Unexpected error validating {dataset_name} item at index {i}: {e}")
                    continue
            
            validated_results[dataset_name] = validated_items
            logger.info(f"Validated {len(validated_items)} out of {len(items)} {dataset_name} items")
        
        return validated_results
        
    except Exception as e:
        logger.error(f"Error in validate_enhanced_data: {e}")
        return {"courses": [], "certificates": [], "jobs": [], "ebooks": []}

def remove_duplicates_from_validated_data(validated_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Remove duplicates from validated data based on titles
    """
    for dataset_name, items in validated_data.items():
        if not items:
            continue
            
        unique_items = []
        seen_titles = set()
        
        for item in items:
            # Get title for uniqueness check
            title = (item.get('title') or 
                    item.get('page_title') or 
                    item.get('course_name') or 
                    str(hash(frozenset(item.items()))))
            
            if title not in seen_titles:
                seen_titles.add(title)
                unique_items.append(item)
            else:
                logger.debug(f"Skipping duplicate {dataset_name} item: {title}")
        
        validated_data[dataset_name] = unique_items
        if len(items) > len(unique_items):
            logger.info(f"Removed {len(items) - len(unique_items)} duplicates from {dataset_name}")
    
    return validated_data

def fix_common_json_issues(json_str: str) -> str:
    """
    Fix common JSON formatting issues that LLMs might introduce
    """
    # Remove any markdown code block markers
    json_str = re.sub(r'```json\s*', '', json_str)
    json_str = re.sub(r'```\s*$', '', json_str)
    
    # Fix trailing commas (common LLM issue)
    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
    
    # Fix missing commas between objects (try to detect and fix)
    json_str = re.sub(r'}\s*{', '},{', json_str)
    json_str = re.sub(r']\s*\[', '],[', json_str)
    
    # Remove any leading/trailing whitespace and newlines
    json_str = json_str.strip()
    
    return json_str

def extract_and_parse_json_robust(content: str) -> Dict[str, Any]:
    """
    More robust JSON extraction and parsing with multiple fallback strategies
    """
    try:
        # Strategy 1: Look for complete JSON object
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            json_str = fix_common_json_issues(json_str)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse complete JSON: {e}")
        
        # Strategy 2: Try to extract individual dataset sections
        datasets = {}
        
        # Look for individual dataset patterns
        dataset_patterns = {
            'courses': r'"courses"\s*:\s*\[(.*?)\]',
            'certificates': r'"certificates"\s*:\s*\[(.*?)\]',
            'jobs': r'"jobs"\s*:\s*\[(.*?)\]',
            'ebooks': r'"ebooks"\s*:\s*\[(.*?)\]'
        }
        
        for dataset_name, pattern in dataset_patterns.items():
            match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
            if match:
                try:
                    # Try to parse individual dataset
                    dataset_json = f'[{match.group(1)}]'
                    dataset_json = fix_common_json_issues(dataset_json)
                    datasets[dataset_name] = json.loads(dataset_json)
                    logger.info(f"Successfully extracted {dataset_name} dataset")
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse {dataset_name} dataset, will try object-by-object")
                    # Strategy 3: Parse individual objects within this dataset
                    datasets[dataset_name] = parse_objects_individually(match.group(1))
        
        if datasets:
            return datasets
            
    except Exception as e:
        logger.error(f"Error in JSON extraction: {e}")
    
    # If all strategies fail, return empty structure
    logger.warning("All JSON parsing strategies failed, returning empty structure")
    return {"courses": [], "certificates": [], "jobs": [], "ebooks": []}

def parse_objects_individually(dataset_content: str) -> List[Dict]:
    """
    Parse individual JSON objects from a dataset section
    """
    objects = []
    
    # Split by likely object boundaries and try to parse each
    potential_objects = re.split(r'},\s*{', dataset_content)
    
    for i, obj_str in enumerate(potential_objects):
        try:
            # Add back the braces that were removed by split
            if i == 0:
                obj_str = obj_str.rstrip(',').rstrip('}') + '}'
            elif i == len(potential_objects) - 1:
                obj_str = '{' + obj_str.lstrip('{')
            else:
                obj_str = '{' + obj_str + '}'
            
            obj_str = fix_common_json_issues(obj_str)
            parsed_obj = json.loads(obj_str)
            objects.append(parsed_obj)
            
        except json.JSONDecodeError:
            # Try to salvage partial data using regex
            salvaged_obj = salvage_object_data(obj_str)
            if salvaged_obj:
                objects.append(salvaged_obj)
    
    return objects

def salvage_object_data(obj_str: str) -> Optional[Dict]:
    """
    Try to salvage data from malformed JSON object using regex
    """
    try:
        obj = {}
        
        # Common field patterns
        field_patterns = {
            'title': r'"title"\s*:\s*"([^"]*)"',
            'course_name': r'"course_name"\s*:\s*"([^"]*)"',
            'page_title': r'"page_title"\s*:\s*"([^"]*)"',
            'description': r'"description"\s*:\s*"([^"]*)"',
            'page_description': r'"page_description"\s*:\s*"([^"]*)"',
            'fee_detail': r'"fee_detail"\s*:\s*"([^"]*)"',
            'learning_term_details': r'"learning_term_details"\s*:\s*"([^"]*)"',
            'course_highlight': r'"course_highlight"\s*:\s*"([^"]*)"',
            'job_details': r'"job_details"\s*:\s*"([^"]*)"',
            'requirements': r'"requirements"\s*:\s*"([^"]*)"',
            'benefits': r'"benefits"\s*:\s*"([^"]*)"',
            'company': r'"company"\s*:\s*"([^"]*)"',
            'location': r'"location"\s*:\s*"([^"]*)"',
            'salary': r'"salary"\s*:\s*"([^"]*)"',
            'author': r'"author"\s*:\s*"([^"]*)"',
            'pages': r'"pages"\s*:\s*"([^"]*)"',
            'format': r'"format"\s*:\s*"([^"]*)"',
            'topics': r'"topics"\s*:\s*"([^"]*)"',
            'level': r'"level"\s*:\s*"([^"]*)"'
        }
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, obj_str, re.IGNORECASE)
            if match:
                obj[field] = match.group(1)
        
        # Only return if we found at least some key fields
        if obj and ('title' in obj or 'course_name' in obj or 'page_title' in obj):
            logger.info(f"Salvaged object with fields: {list(obj.keys())}")
            return obj
            
    except Exception as e:
        logger.debug(f"Failed to salvage object: {e}")
    
    return None

def validate_item_with_partial_data(item: Dict, model_class: BaseModel, dataset_name: str, index: int) -> Optional[Dict]:
    """
    Validate item and try to salvage partial data if validation fails
    """
    try:
        # First try normal validation
        validated_item = model_class(**item)
        return validated_item.dict()
        
    except ValidationError as e:
        # Log specific validation errors for debugging
        error_details = []
        for error in e.errors():
            field = error.get('loc', ['unknown'])[0]
            msg = error.get('msg', 'validation error')
            error_details.append(f"{field}: {msg}")
        
        logger.info(f"Validation failed for {dataset_name}[{index}] - {', '.join(error_details[:3])}{'...' if len(error_details) > 3 else ''}")
        logger.debug(f"Attempting partial salvage for {dataset_name}[{index}]")
        
        # Try to create a minimal valid object with available data
        salvaged_data = {}
        
        # Get model fields information
        model_fields = model_class.__fields__
        
        # First pass: Copy valid existing data
        for field_name, field_info in model_fields.items():
            if field_name in item:
                value = item[field_name]
                # Clean and validate the value
                if isinstance(value, str):
                    value = value.strip()
                    if value:  # Only use non-empty strings
                        salvaged_data[field_name] = value
                elif value is not None:
                    salvaged_data[field_name] = str(value).strip()
        
        # Second pass: Fill missing required fields
        for field_name, field_info in model_fields.items():
            if field_name not in salvaged_data:
                if not field_info.required:
                    # Optional field - use default or empty string
                    default_val = getattr(field_info, 'default', "")
                    salvaged_data[field_name] = default_val if default_val is not ... else ""
                else:
                    # Required field missing - try to generate or derive
                    if field_name in ['title', 'course_name', 'page_title']:
                        # Try to find any title-like field
                        potential_title = (
                            item.get('title') or 
                            item.get('course_name') or 
                            item.get('page_title') or 
                            item.get('name') or
                            f"Enhanced {dataset_name.rstrip('s').title()}"
                        )
                        salvaged_data[field_name] = str(potential_title).strip()
                    elif field_name in ['description', 'page_description']:
                        # Generate basic description
                        title = salvaged_data.get('title') or salvaged_data.get('course_name') or salvaged_data.get('page_title', '')
                        salvaged_data[field_name] = f"Learn about {title}. This comprehensive program covers essential topics and practical skills."
                    elif 'detail' in field_name.lower():
                        salvaged_data[field_name] = "Details will be provided upon enrollment."
                    elif 'requirement' in field_name.lower():
                        salvaged_data[field_name] = "Basic knowledge recommended."
                    elif 'benefit' in field_name.lower():
                        salvaged_data[field_name] = "Comprehensive learning experience with practical applications."
                    else:
                        salvaged_data[field_name] = "Information available upon request."
        
        try:
            # Try to validate the salvaged data
            validated_item = model_class(**salvaged_data)
            logger.info(f"✅ Successfully salvaged {dataset_name}[{index}] with {len([k for k, v in salvaged_data.items() if v and v.strip()])} valid fields")
            return validated_item.dict()
            
        except ValidationError as salvage_error:
            # Log what specifically failed in salvage attempt
            salvage_errors = [f"{err.get('loc', ['unknown'])[0]}: {err.get('msg', 'error')}" for err in salvage_error.errors()]
            logger.warning(f"❌ Could not salvage {dataset_name}[{index}] - {', '.join(salvage_errors[:2])}")
            return None
    
    except Exception as e:
        logger.error(f"Unexpected error validating {dataset_name}[{index}]: {e}")
        return None

def validate_enhanced_data_robust(enhanced_content: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Enhanced validation with robust JSON parsing and partial data recovery
    """
    try:
        # Use robust JSON extraction
        enhanced_all_data = extract_and_parse_json_robust(enhanced_content)
        
        if not isinstance(enhanced_all_data, dict):
            logger.warning("Enhanced data is not a dictionary")
            return {"courses": [], "certificates": [], "jobs": [], "ebooks": []}
        
        # Initialize result structure
        validated_results = {
            "courses": [],
            "certificates": [],
            "jobs": [],
            "ebooks": []
        }
        
        # Model mapping
        model_mapping = {
            "courses": Course,
            "certificates": Certificate,
            "jobs": Job,
            "ebooks": Ebooks
        }
        
        # Track validation statistics
        validation_stats = {
            "total_items": 0,
            "valid_items": 0,
            "salvaged_items": 0,
            "failed_items": 0
        }
        
        # Validate each dataset with partial recovery
        for dataset_name, items in enhanced_all_data.items():
            if dataset_name not in model_mapping:
                logger.warning(f"Unknown dataset type: {dataset_name}")
                continue
                
            if not isinstance(items, list):
                logger.warning(f"Dataset '{dataset_name}' is not a list")
                continue
            
            model_class = model_mapping[dataset_name]
            validated_items = []
            dataset_stats = {"total": len(items), "valid": 0, "salvaged": 0, "failed": 0}
            
            logger.info(f"Processing {len(items)} {dataset_name} items...")
            
            for i, item in enumerate(items):
                validation_stats["total_items"] += 1
                dataset_stats["total"] += 1
                
                if not isinstance(item, dict):
                    logger.warning(f"Item {i} in {dataset_name} is not a dictionary")
                    dataset_stats["failed"] += 1
                    validation_stats["failed_items"] += 1
                    continue
                
                # Check if item needs salvaging by trying validation first
                try:
                    model_class(**item)
                    # If we get here, validation passed
                    validated_item = validate_item_with_partial_data(item, model_class, dataset_name, i)
                    if validated_item:
                        validated_items.append(validated_item)
                        dataset_stats["valid"] += 1
                        validation_stats["valid_items"] += 1
                except ValidationError:
                    # Item needs salvaging
                    validated_item = validate_item_with_partial_data(item, model_class, dataset_name, i)
                    if validated_item:
                        validated_items.append(validated_item)
                        dataset_stats["salvaged"] += 1
                        validation_stats["salvaged_items"] += 1
                    else:
                        dataset_stats["failed"] += 1
                        validation_stats["failed_items"] += 1
            
            validated_results[dataset_name] = validated_items
            
            # Log dataset summary
            logger.info(f"📊 {dataset_name.upper()} Summary: {dataset_stats['valid']} valid, {dataset_stats['salvaged']} salvaged, {dataset_stats['failed']} failed out of {dataset_stats['total']} total")
        
        # Log overall summary
        logger.info(f"🎯 OVERALL Summary: {validation_stats['valid_items']} valid, {validation_stats['salvaged_items']} salvaged, {validation_stats['failed_items']} failed out of {validation_stats['total_items']} total items")
        
        return validated_results
        
    except Exception as e:
        logger.error(f"Error in validate_enhanced_data_robust: {e}")
        return {"courses": [], "certificates": [], "jobs": [], "ebooks": []}