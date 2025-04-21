import asyncio, websockets, time
from gameAI import FootsiesPredictor

import tensorflow as tf

# message handler for connected clients
async def message_handler(websocket):

    try:
        # this block is where message recieving and processing happens
        async for message in websocket:

            start = time.perf_counter()
            normData = footsiesAI.prepareData(message)
            print(f"PrepareData: {(time.perf_counter() - start)*1000:.2f}ms")

            start = time.perf_counter()
            footsiesAI.addFrame(normData)
            print(f"AddFrame: {(time.perf_counter() - start)*1000:.2f}ms")

            start = time.perf_counter()
            output = footsiesAI.predict()
            print(f"Predict: {(time.perf_counter() - start)*1000:.2f}ms")

            await websocket.send(str(output))
    
    # exception handling
    except Exception as e:
        print(f"Error: {e}.")
        input()

    finally:
        print("Connection closed.")


# start the WebSocket server
async def control_server():

    server = await websockets.serve(message_handler, "localhost", 8677)
    print("WebSocket server started on ws://localhost:8677")
    await server.wait_closed()


# run the server
if __name__ == "__main__":

    try:

        footsiesAI = FootsiesPredictor(modelPath="peak.keras",
                                       sequenceLength=20,
                                       features=46)
        
        
    except Exception as e:
        print(f"Failed to initialise network: {e}")
        input()

    try:
        asyncio.run(control_server())

    except Exception as e:
        print(f"Error: {e}")
        input()