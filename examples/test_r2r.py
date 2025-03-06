#from r2r import R2RClient
R2R_BASE_URL = 'http://host.docker.internal:7272'
R2R_USERNAME = 'admin@example.com'
R2R_PASSWORD = 'change_me_immediately'

from typing import Optional, Any

from r2r import R2RClient
import json

class R2R:
    client: R2RClient

    def init(self):
        self.client = R2RClient(R2R_BASE_URL)
        self.auth_enabled = R2R_USERNAME and R2R_PASSWORD
        #self._login()
        #self._check_login()
        self.ingest_file(file_path = 'test_r2r.txt', metadata = None)
        self.search("Introduce AWS Health")

    def ingest_file(self, file_path: str, metadata: Optional[dict]):
        self._check_login()
        try:
            ingest_response = self.client.documents.create(
                file_path=file_path,
                ingestion_mode = "custom",
                run_with_orchestration=False,
                ingestion_config = {
                    "chunk_size": 512,
                }
            )
            results = ingest_response.model_dump().get('results')
        except Exception as e:
            print(f'Insertion failed, error: {e}')
            return None
        else:
            print(results.get('message'))
            return results

    def search_chunks(self, query: str, filters: dict[str, Any]):
        search_response = self.client.chunks.search(
            query=query,
            search_settings={
                "use_hybrid_search": True,
                "filters": filters,
                "search_limit": 5,
            },
        )
        results = search_response.model_dump().get('results')

        # Format the output
        str_mapping = {'id': True, 'metadata': False, 'title': False, 'version': False, 'summary': False}
        temp_results = []
        for result in results:
            document_id = str(result.get("document_id"))
            chunk_id = str(result.get("id"))
            response = self.client.chunks.retrieve(id=chunk_id)
            text = response.model_dump().get('results').get("text")
            temp_result = {"id": document_id, "summary": text}
            temp_results.append(temp_result)

        return temp_results

    def search_documents(self, query: str, filters: dict[str, Any]):
        try:
            search_response = self.client.documents.search(
                query=query,
                search_mode = 'custom',
                search_settings={
                    "filters": filters,
                    "search_limit": 10,
                    "do_hybrid_search": True
                },
            )
            results = search_response.model_dump().get('results')
        except Exception as e:
            print(f'Search in the VDB failed, error: {e}')
            return None
        else:
            print('Search successfully, results0 score:', results[0].get('metadata').get('search_score'))
            print('Search successfully, vector length:', len(results[0].get('summary_embedding')))
            print('Document id:', results[0].get('id'))
            print('Document text:', results[0].get('summary'))
            return results

    def search(self, query: str, filters: dict[str, Any]):
        self._check_login()
        try:
            results = self.search_chunks(query, filters)
        except Exception as e:
            print(f'Search in the VDB failed, error: {e}')
            return None
        else:
            return results

    def _login(self):
        if not self.auth_enabled:
            return
        self.client.users.login(R2R_USERNAME, R2R_PASSWORD)

    def _check_login(self):
        if not self.auth_enabled:
            return
        if verify_jwt_expiration(self.client.access_token):
            return
        if verify_jwt_expiration(self.client._refresh_token):
            self.client.refresh_access_token()
        else:
            self._login()


r2r = R2R()

r2r.init()
