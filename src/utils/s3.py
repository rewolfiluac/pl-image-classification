from typing import Any, Callable
import gorilla
import fsspec


def apply_gorrila(function: Callable, module: Any):
    """Overriding a function using a gorilla patch.

    Args:
        function (Callable): Override function
        module (Any): Function caller module
    """
    patch = gorilla.Patch(
        module, function.__name__, function, settings=gorilla.Settings(allow_hit=True)
    )
    gorilla.apply(patch)


def setup_endpoint(endpoint_url: str):
    """Specify the endpoint to be saved to S3.

    Args:
        endpoint_url (str): endpoint url
    """

    def filesystem(protocol, **storage_options):
        if protocol == "s3":
            storage_options["client_kwargs"] = {
                "endpoint_url": endpoint_url,
            }
        original = gorilla.get_original_attribute(fsspec, "filesystem")
        return original(protocol, **storage_options)

    def open_files(*args, **kwargs):
        kwargs["client_kwargs"] = {
            "endpoint_url": endpoint_url,
        }
        original = gorilla.get_original_attribute(fsspec.core, "open_files")
        return original(*args, **kwargs)

    apply_gorrila(filesystem, fsspec)
    apply_gorrila(open_files, fsspec.core)
