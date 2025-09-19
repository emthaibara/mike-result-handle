import os

work_space_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
assets_path = os.path.join(work_space_path, 'assets')
generated_path = os.path.join(assets_path, 'generated')
cases_json_path = os.path.join(generated_path, 'cases.json')