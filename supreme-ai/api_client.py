import aiohttp
from typing import Dict, Optional
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimulationAPI:
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_metrics_update(
        self,
        solution: pd.DataFrame,
        solution_without_tariff: pd.DataFrame,
        solution_without_rl: pd.DataFrame,
        quarter_str: str,
        targets: Dict[str, float]
    ):
        """Send metrics update to the API"""
        def create_metrics(df: pd.DataFrame) -> Dict:
            current = df.loc[quarter_str]
            previous = df.shift(1).loc[quarter_str]
            
            return {
                "quarter": quarter_str,
                "metrics": {
                    'hggdp': float(current['hggdp']),
                    'xgdpn': float(current['xgdpn']),
                    'xgdp': float(current['xgdp']),
                    'tpn': float(current['tpn']),
                    'tcin': float(current['tcin']),
                    'trp': float(current['trp']),
                    'trci': float(current['trci']),
                    'gtrt': float(current['gtrt']),
                    'egfet': float(current['egfet']),
                    'frs10': float(current['frs10']),
                    'pcpi': float(current['pcpi']),
                    'lur': float(current['lur']),
                    'gfdbtn': float(current['gfdbtn']),
                    'emn': float(current['emn']),
                    'exn': float(current['exn'])
                },
                "previous_metrics": {
                    'hggdp': float(previous['hggdp']),
                    'xgdpn': float(previous['xgdpn']),
                    'xgdp': float(previous['xgdp']),
                    'tpn': float(previous['tpn']),
                    'tcin': float(previous['tcin']),
                    'trp': float(previous['trp']),
                    'trci': float(previous['trci']),
                    'gtrt': float(previous['gtrt']),
                    'egfet': float(previous['egfet']),
                    'frs10': float(previous['frs10']),
                    'pcpi': float(previous['pcpi']),
                    'lur': float(previous['lur']),
                    'gfdbtn': float(previous['gfdbtn']),
                    'emn': float(previous['emn']),
                    'exn': float(previous['exn'])
                },
                "targets": targets
            }

        metrics_data = {
            "rl_decision": create_metrics(solution),
            "without_tariff": create_metrics(solution_without_tariff),
            "base_simulation": create_metrics(solution_without_rl)
        }

        try:
            async with self.session.post(
                f"{self.api_url}/simulation/update",
                json=metrics_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Error sending metrics: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Failed to send metrics: {str(e)}")
            return None
