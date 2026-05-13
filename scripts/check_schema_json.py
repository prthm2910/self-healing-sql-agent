import json
from src.workflow.schema import GuardianOutput

schema = GuardianOutput.model_json_schema()
print(json.dumps(schema, indent=2))
