from gameAI import FootsiesPredictor
import tensorflow as tf
import asyncio, websockets, time, sys

shutdownEvent = asyncio.Event()

async def messageHandler(websocket):
    '''
    This function handles all messages sent by the client in an asynchronous
        way
    The server is opened on port 8677 and this function is run asynchronously
        to receieve and then process all messages.
    Argument type: websocket
    '''

    try:
        # this block is where message recieving and processing happens
        async for message in websocket:

            start = time.perf_counter()
            footsiesAI.prepareData(message)
            print(f"PrepareData: {(time.perf_counter() - start)*1000:.2f}ms")

            start = time.perf_counter()
            output = footsiesAI.predict()
            print(f"Predict: {(time.perf_counter() - start)*1000:.2f}ms")

            await websocket.send(str(output))
    
    # exception handling
    except websockets.ConnectionClosed:
        print("Client disconnected.")

    except Exception as e:
        print(f"Error: {e}.")

    finally:
        print("Connection closed.")
        shutdownEvent.set()

# start the WebSocket server
async def control_server():

    server = await websockets.serve(messageHandler, 
                                    "localhost", 
                                    8677,
                                    ping_interval=5,
                                    ping_timeout=10)
    
    print("WebSocket server started on ws://localhost:8677")
    await shutdownEvent.wait()
    print("\nServer shutting down...")
    # time.sleep(3)
    input()

# run the server
if __name__ == "__main__":

    try:

        # creates instance of the predictor class
        footsiesAI = FootsiesPredictor(modelPath="FootsiesNeuralNetwork.keras",
                                       sequenceLength=20,
                                       features=34,
                                       predictIntervals=4)
        
        asyncio.run(control_server())

    except Exception as e:
        print(f"Failed initialise server and/or network: {e}")
        sys.exit(1)