from typing import Any
import gorilla
import fsspec

def apply_gorrila(function: Any, module: Any):
    patch = gorilla.Patch(
        module, 
        function.__name__, 
        function, 
        settings=gorilla.Settings(allow_hit=True))
    gorilla.apply(patch)

def setup_endpoint(endpoint_url: str):
    def filesystem(protocol, **storage_options):
        if protocol == "s3":
            storage_options["client_kwargs"] = {
                "endpoint_url": endpoint_url,
            }
        original = gorilla.get_original_attribute(fsspec, 'filesystem')
        return original(protocol, **storage_options)
    def open_files(*args, **kwargs):
        kwargs["client_kwargs"] = {
                "endpoint_url": endpoint_url,
        }
        original = gorilla.get_original_attribute(fsspec.core, 'open_files')
        return original(*args, **kwargs)
    apply_gorrila(filesystem, fsspec)
    apply_gorrila(open_files, fsspec.core)
