import os
import sys
import yaml
import importlib
import importlib.util
from pathlib import Path


class ConfigManager:
    def __init__(self, project_root=None):
        if project_root is None:
            self.project_root = self._find_project_root()
        else:
            self.project_root = Path(project_root)
        self.config_dir = self.project_root / 'config'
        self.main_config = self.load_main_config()
        self.local_configs = {}
        self.source_dirs = {
            'mcr-agent': '/content/mcr-agent',
            'alfred': '/content/alfred'
        }

    def _find_project_root(self):
        """Find the root of the project automatically"""
        current = Path.cwd()
        while current.parent != current:
            if (current / "requirements.txt").exists() or \
                    (current / "setup.py").exists() or \
                    (current / ".git").exists():
                return current
            current = current.parent
        return current

    def load_main_config(self):
        """Load the main config.yaml file"""
        config_path = self.config_dir / 'config.yaml'
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def load_local_config(self, module_path):
        """Load local config for a specific module"""
        local_config_file = Path(module_path) / "local_override.yaml"
        if local_config_file.exists():
            with open(local_config_file, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        return {}

    def get_override(self, module_name):
        """Get override configuration for a specific module"""
        # Check main config first
        if 'overrides' in self.main_config and module_name in self.main_config['overrides']:
            return self.main_config['overrides'][module_name]

        # Check local config if available
        module_path = module_name.replace('.', '/')
        local_config = self.load_local_config(module_path)
        if 'overrides' in local_config and module_name in local_config['overrides']:
            return local_config['overrides'][module_name]

        return None

    def import_with_override(self, module_name):
        """Import a module with potential overrides"""
        override = self.get_override(module_name)

        if override and override.get('action') == 'override':
            source = override.get('source')
            path = override.get('path')

            if source in self.source_dirs and path:
                # Build the full path to the module
                full_path = os.path.join(self.source_dirs[source], path)

                if os.path.exists(full_path):
                    # Use importlib to load from the specific path
                    spec = importlib.util.spec_from_file_location(module_name, full_path)
                    if spec:
                        module = importlib.util.module_from_spec(spec)
                        sys.modules[module_name] = module
                        spec.loader.exec_module(module)
                        return module

        # Fall back to regular import
        return importlib.import_module(module_name)


# Initialize config manager
config_manager = ConfigManager()