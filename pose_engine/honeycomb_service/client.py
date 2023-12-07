from functools import lru_cache
import os

import honeycomb_io
import minimal_honeycomb


class HoneycombCachingClient:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __init__(
        self,
        url=None,
        auth_domain=None,
        auth_client_id=None,
        auth_client_secret=None,
        auth_audience=None,
    ):
        url = (
            os.getenv(
                "HONEYCOMB_URI", "https://honeycomb.api.wildflower-tech.org/graphql"
            )
            if url is None
            else url
        )
        auth_domain = (
            os.getenv(
                "HONEYCOMB_DOMAIN",
                os.getenv("AUTH0_DOMAIN", "wildflowerschools.auth0.com"),
            )
            if auth_domain is None
            else auth_domain
        )
        auth_client_id = (
            os.getenv("HONEYCOMB_CLIENT_ID", os.getenv("AUTH0_CLIENT_ID", None))
            if auth_client_id is None
            else auth_client_id
        )
        auth_client_secret = (
            os.getenv("HONEYCOMB_CLIENT_SECRET", os.getenv("AUTH0_CLIENT_SECRET", None))
            if auth_client_secret is None
            else auth_client_secret
        )
        auth_audience = (
            os.getenv(
                "HONEYCOMB_AUDIENCE", os.getenv("API_AUDIENCE", "wildflower-tech.org")
            )
            if auth_audience is None
            else auth_audience
        )

        if auth_client_id is None:
            raise ValueError("HONEYCOMB_CLIENT_ID (or AUTH0_CLIENT_ID) is required")
        if auth_client_secret is None:
            raise ValueError(
                "HONEYCOMB_CLIENT_SECRET (or AUTH0_CLIENT_SECRET) is required"
            )

        token_uri = os.getenv(
            "HONEYCOMB_TOKEN_URI", f"https://{auth_domain}/oauth/token"
        )

        self.client: minimal_honeycomb.MinimalHoneycombClient = (
            honeycomb_io.generate_client(
                uri=url,
                token_uri=token_uri,
                audience=auth_audience,
                client_id=auth_client_id,
                client_secret=auth_client_secret,
            )
        )

        self.client_params = {
            "client": self.client,
            "uri": url,
            "token_uri": token_uri,
            "audience": auth_audience,
            "client_id": auth_client_id,
            "client_secret": auth_client_secret,
        }

    @lru_cache(maxsize=50)
    def fetch_camera_devices(
        self,
        environment_id=None,
        environment_name=None,
        start=None,
        end=None,
        chunk_size=200,
    ):
        return honeycomb_io.fetch_devices(
            device_types=honeycomb_io.DEFAULT_CAMERA_DEVICE_TYPES,
            environment_id=environment_id,
            environment_name=environment_name,
            start=start,
            end=end,
            output_format="dataframe",
            chunk_size=chunk_size,
            **self.client_params,
        )

    @lru_cache()
    def fetch_all_environments(self):
        return honeycomb_io.fetch_all_environments(
            output_format="dataframe", **self.client_params
        )

    @lru_cache()
    def fetch_environment(self, environment):
        df_all_environments = self.fetch_all_environments()
        df_all_environments = df_all_environments.reset_index()
        for _, environment_row in df_all_environments.iterrows():
            if environment_row["environment_id"] == str(environment) or environment_row[
                "environment_name"
            ] == str(environment):
                return environment_row.to_dict()

        return None
