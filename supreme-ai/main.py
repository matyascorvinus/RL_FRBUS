from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict
from datetime import datetime
import json

from models import EconomicMetrics, SimulationComparison

app = FastAPI(
    title="Economic FRB/US Simulation API",
    description="Real-time economic metrics streaming API",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except WebSocketDisconnect:
                await self.disconnect(connection)

manager = ConnectionManager()

# Store latest metrics in memory (you might want to use Redis in production)
latest_metrics: Dict[str, SimulationComparison] = {}

@app.websocket("/ws/metrics")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # Send latest metrics immediately upon connection
        if latest_metrics:
            print(f"Sending latest metrics: {latest_metrics}")
            await websocket.send_json(latest_metrics)
        
        while True:
            await websocket.receive_text()  # Keep connection alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/simulation/update")
async def update_simulation(data: SimulationComparison):
    """
    Receive new simulation data and broadcast to all connected clients
    """
    try:
        # Store latest metrics
        latest_metrics[data.rl_decision.quarter] = data.dict()
        # Broadcast to all connected clients
        await manager.broadcast(data.dict())
        
        return {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "clients": len(manager.active_connections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/simulation/latest")
async def get_latest_metrics():
    """
    Get the most recent metrics
    """
    if not latest_metrics:
        raise HTTPException(status_code=404, detail="No metrics available")
    return latest_metrics

@app.get("/simulation/quarters/{quarter}")
async def get_quarter_metrics(quarter: str):
    """
    Get metrics for a specific quarter
    """
    if quarter not in latest_metrics:
        raise HTTPException(status_code=404, detail=f"No data for quarter {quarter}")
    return latest_metrics[quarter]