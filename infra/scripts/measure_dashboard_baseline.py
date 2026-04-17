import asyncio
import time
import argparse
import httpx

DEFAULT_API_BASE = "http://localhost:8000/api/v1"

async def measure_latency(client: httpx.AsyncClient, api_base: str, endpoint: str, name: str):
    url = f"{api_base.rstrip('/')}{endpoint}"
    start_time = time.perf_counter()
    try:
        response = await client.get(url)
        end_time = time.perf_counter()
        latency = end_time - start_time
        print(f"{name} ({endpoint}): {latency:.4f}s - Status: {response.status_code}")
        return latency
    except Exception as e:
        print(f"Error measuring {name}: {e}")
        return None


async def main(api_base: str, repeat: int):
    print("Measuring baseline dashboard latencies...")

    async with httpx.AsyncClient() as client:
        await measure_latency(client, api_base, "/dashboard/stats", "Manager Stats")

        resp = await client.get(f"{api_base.rstrip('/')}/agents")
        if resp.status_code == 200:
            agents = resp.json()
            if agents:
                agent_id = agents[0]["id"]
                await measure_latency(client, api_base, f"/agents/{agent_id}", f"Agent Profile ({agents[0]['name']})")

        print(f"\nStarting {repeat} repeated measurements for Manager Stats...")
        latencies = []
        for i in range(repeat):
            latency = await measure_latency(client, api_base, "/dashboard/stats", f"Run {i + 1}")
            if latency:
                latencies.append(latency)

    if latencies:
        print(f"\nAverage Manager Stats Latency: {sum(latencies)/len(latencies):.4f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Measure dashboard API baseline latencies.")
    parser.add_argument("--api-base", default=DEFAULT_API_BASE, help="Backend API base URL.")
    parser.add_argument("--repeat", type=int, default=5, help="How many repeated dashboard calls to run.")
    args = parser.parse_args()

    try:
        asyncio.run(main(args.api_base, args.repeat))
    except Exception as e:
        print(f"Failed to run measurements: {e}")
