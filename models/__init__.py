original_import_module = importlib.import_module

def patched_import_module(name, package=None):
    try:
        return config_manager.import_with_override(name)
    except:
        return original_import_module(name, package)

importlib.import_module = patched_import_module
