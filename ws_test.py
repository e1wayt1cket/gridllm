import asyncio
import urllib.request

import websockets


def test_http():
    url = 'http://localhost:8001/'
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            data = r.read(200)
            print('HTTP OK, bytes:', len(data))
    except Exception as e:
        print('HTTP error:', e)


async def ws_client():
    uri = 'ws://localhost:8001/ws'
    try:
        async with websockets.connect(uri) as ws:
            # 发送保持连接的文本，服务端不特别响应，但会维持连接
            await ws.send('ping')
            try:
                msg = await asyncio.wait_for(ws.recv(), timeout=8)
                print('WS RECV:', msg)
            except asyncio.TimeoutError:
                print('WS: no message received within timeout')
    except Exception as e:
        print('WS connect error:', e)


if __name__ == '__main__':
    test_http()
    asyncio.run(ws_client())
