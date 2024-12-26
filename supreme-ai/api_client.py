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
                "quarter": quarter_str.upper(),
                "metrics": {
                    'hggdp': float(current['hggdp']),
                    'xgdpn': float(current['xgdpn']),
                    'xgdp': float(current['xgdp']),
                    'tpn': float(current['tpn']),
                    'tcin': float(current['tcin']),
                    'trptx': float(current['trptx']),
                    'trcit': float(current['trcit']),
                    'gtrt': float(current['gtrt']),
                    'egfe': float(current['egfe']),
                    'rff': float(current['rff']),
                    'pcpi': float(current['pcpi']),
                    'lur': float(current['lur']),
                    'gfdbtn': float(current['gfdbtn']),
                    'emn': float(current['emn']),
                    'exn': float(current['exn']),
                    'gtn': float(current['gtn']),
                    'gfsrpn': float(current['gfsrpn'])
                },
                "previous_metrics": {
                    'hggdp': float(previous['hggdp']),
                    'xgdpn': float(previous['xgdpn']),
                    'xgdp': float(previous['xgdp']),
                    'tpn': float(previous['tpn']),
                    'tcin': float(previous['tcin']),
                    'trptx': float(previous['trptx']),
                    'trcit': float(previous['trcit']),
                    'gtrt': float(previous['gtrt']),
                    'egfe': float(previous['egfe']),
                    'rff': float(previous['rff']),
                    'pcpi': float(previous['pcpi']),
                    'lur': float(previous['lur']),
                    'gfdbtn': float(previous['gfdbtn']),
                    'emn': float(previous['emn']),
                    'exn': float(previous['exn']),
                    'gtn': float(previous['gtn']),
                    'gfsrpn': float(previous['gfsrpn'])
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
