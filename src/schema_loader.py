import json
from pathlib import Path
from typing import Dict, List, Any


class SchemaLoadError(Exception):
    pass

class SchemaLoader:
    
    def __init__(self, schema_dir: str = "data/schemas"):
        self.schema_dir = Path(schema_dir)
        self.schemas: Dict[str, Any] = {}
        self._available_schemas = None
    
    @property
    def available_schemas(self) -> List[str]:
        if self._available_schemas is None:
            # Find all .json files in the schema directory and extract their names
            if not self.schema_dir.exists():
                self._available_schemas = []
            else:
                self._available_schemas = [
                    f.stem for f in self.schema_dir.glob("*.json")
                ]
        
        return self._available_schemas
    
    def get_schema(self, bank_name: str) -> Dict[str, Any]:
        # Return cached schema if available
        if bank_name in self.schemas:
            return self.schemas[bank_name]
        
        # Construct the file path
        schema_path = self.schema_dir / f"{bank_name}.json"
        
        # Check if the file exists
        if not schema_path.exists():
            raise SchemaLoadError(f"Schema file for '{bank_name}' not found at {schema_path}")
        
        # Load and parse the schema
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            
            # Validate schema structure at a basic level
            if "columns" not in schema and "sections" not in schema:
                raise SchemaLoadError(
                    f"Invalid schema format for '{bank_name}': missing both 'columns' and 'sections'"
                )
            
            # Validate column structure in the schema
            self._validate_columns(schema, bank_name)
            
            # Cache the schema for future use
            self.schemas[bank_name] = schema
            return schema
            
        except json.JSONDecodeError as e:
            raise SchemaLoadError(f"Failed to parse schema for '{bank_name}': {str(e)}")
        except Exception as e:
            raise SchemaLoadError(f"Error loading schema for '{bank_name}': {str(e)}")
    
    def _validate_columns(self, schema: Dict[str, Any], bank_name: str) -> None:
        if "columns" in schema:
            self._check_column_fields(schema["columns"], bank_name)
        
        if "sections" in schema:
            for i, section in enumerate(schema["sections"]):
                if "columns" in section:
                    self._check_column_fields(section["columns"], f"{bank_name} section {i}")
    
    def _check_column_fields(self, columns: List[Dict[str, Any]], context: str) -> None:
        for i, col in enumerate(columns):
            if not isinstance(col, dict):
                raise SchemaLoadError(f"Column {i} in {context} is not a dictionary")
            
            if "key" not in col:
                raise SchemaLoadError(f"Column {i} in {context} is missing 'key' field")
            
            if "name" not in col:
                raise SchemaLoadError(f"Column {i} in {context} is missing 'name' field")
    
    def get_columns(self, bank_name: str) -> List[Dict[str, Any]]:
        schema = self.get_schema(bank_name)
        
        # Case 1: Direct columns at the top level
        if "columns" in schema:
            return schema["columns"]
        
        # Case 2: Columns within sections
        if "sections" in schema:
            # Combine columns from all sections
            all_columns = []
            for section in schema["sections"]:
                if "columns" in section:
                    all_columns.extend(section["columns"])
            
            if not all_columns:
                raise SchemaLoadError(
                    f"No column definitions found in sections for '{bank_name}'"
                )
            
            return all_columns
        
        # This shouldn't happen because of validation in get_schema()
        raise SchemaLoadError(f"No column definitions found for '{bank_name}'")
    
    def get_sections(self, bank_name: str) -> List[Dict[str, Any]]:
        schema = self.get_schema(bank_name)
        return schema.get("sections", [])
    
    def get_table_markers(self, bank_name: str) -> Dict[str, str]:
        schema = self.get_schema(bank_name)
        return schema.get("transaction_table_markers", {})


# Create a singleton instance for easy importing
_loader = SchemaLoader()

# Expose key functions at the module level
def get_columns(bank_name: str) -> List[Dict[str, Any]]:
    """Get column definitions for a specific bank schema."""
    return _loader.get_columns(bank_name)

def get_schema(bank_name: str) -> Dict[str, Any]:
    """Get the full schema for a specific bank."""
    return _loader.get_schema(bank_name)

def get_available_schemas() -> List[str]:
    """Get a list of all available schema names."""
    return _loader.available_schemas

def get_sections(bank_name: str) -> List[Dict[str, Any]]:
    """Get section definitions for a multi-section bank schema."""
    return _loader.get_sections(bank_name)

def get_table_markers(bank_name: str) -> Dict[str, str]:
    """Get transaction table markers for locating tables in documents."""
    return _loader.get_table_markers(bank_name)

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from pathlib import Path
    
    script_path = Path(__file__).resolve()
    
    root_dir = script_path.parent.parent
    
    schema_dir = root_dir / "data" / "schemas"
    print(f"Looking for schemas in: {schema_dir}")
    
    loader = SchemaLoader(schema_dir=schema_dir)
    schemas = loader.available_schemas
    if not schemas:
        print(f"No schemas found in {loader.schema_dir}")
        exit(1)

    print("Found schemas:", schemas)
    for name in schemas:
        try:
            schema = loader.get_schema(name)
            cols   = loader.get_columns(name)
            print(f"\n→ {name}: ✓ loaded ({len(cols)} columns)")
            print("   keys:", list(schema.keys()))
        except Exception as e:
            print(f"\n→ {name}: ❌ {e}")
            exit(1)

    print("\nAll schemas loaded and column-definitions validated.")