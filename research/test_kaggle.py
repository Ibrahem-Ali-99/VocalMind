import httpx, asyncio

KAGGLE = 'https://imani-levorotatory-stingily.ngrok-free.dev'
HEADERS = {'ngrok-skip-browser-warning': 'true'}
TEST_FILE = r'G:\projects\VocalMind\research\voices-examples\DEX_channel_separated_callcenter\2077592167\2077592167_final_stereo.wav'

async def test():
    async with httpx.AsyncClient(timeout=300) as c:
        with open(TEST_FILE, 'rb') as f:
            data = f.read()

        print('=== Emotion Analysis (/predict) ===')
        try:
            r1 = await c.post(f'{KAGGLE}/predict', files={'file': ('test.wav', data, 'audio/wav')}, headers=HEADERS)
            print(f'Status: {r1.status_code}')
            print(f'Result: {r1.text[:300]}...')
        except Exception as e:
            print(f'Error: {e}')

        print('\n=== VAD Segmentation (/split) ===')
        try:
            r2 = await c.post(f'{KAGGLE}/split', files={'file': ('test.wav', data, 'audio/wav')}, headers=HEADERS)
            print(f'Status: {r2.status_code}')
            if r2.status_code == 200:
                print(f"Total Segments: {r2.json().get('total_segments', 0)}")
            else:
                print(f'Result: {r2.text[:300]}...')
        except Exception as e:
            print(f'Error: {e}')

asyncio.run(test())
