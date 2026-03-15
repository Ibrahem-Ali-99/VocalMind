import asyncio
import time
import httpx

API_BASE = "http://localhost:8000/api/v1"

async def measure_latency(endpoint, name):
    async with httpx.AsyncClient() as client:
        start_time = time.perf_counter()
        try:
            response = await client.get(f"{API_BASE}{endpoint}")
            end_time = time.perf_counter()
            latency = end_time - start_time
            print(f"{name} ({endpoint}): {latency:.4f}s - Status: {response.status_code}")
            return latency
        except Exception as e:
            print(f"Error measuring {name}: {e}")
            return None

async def main():
    print("Measuring baseline dashboard latencies...")
    
    # Measure Manager Dashboard Stats
    await measure_latency("/dashboard/stats", "Manager Stats")
    
    # Measure Agent Profiles (assuming IDs from previous file views, or we need to fetch them)
    # Let's fetch agents first
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{API_BASE}/agents")
        if resp.status_code == 200:
            agents = resp.json()
            if agents:
                agent_id = agents[0]['id']
                await measure_latency(f"/agents/{agent_id}", f"Agent Profile ({agents[0]['name']})")
    
    print("\nStarting 5 repeated measurements for Manager Stats...")
    latencies = []
    for i in range(5):
        l = await measure_latency("/dashboard/stats", f"Run {i+1}")
        if l: latencies.append(l)
    
    if latencies:
        print(f"\nAverage Manager Stats Latency: {sum(latencies)/len(latencies):.4f}s")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        print(f"Failed to run measurements: {e}")
