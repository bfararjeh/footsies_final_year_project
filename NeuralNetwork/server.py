import asyncio
import websockets

# Message handler for connected clients
async def message_handler(websocket):

    try:
        # this block is where message recieving and processing happens
        async for message in websocket:

            # currently just printing the recieved message and echoing it back
            print(f"Received message: {message}")
            # await websocket.send(f"Server received: {message}")
    
    # exception handling
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed unexpectedly")
        input()

    except Exception as e:
        print(f"Error: {e}")
        input()

    finally:
        print("Connection closed")
        input()


# start the WebSocket server
async def start_server():

    server = await websockets.serve(message_handler, "localhost", 8677)
    print("WebSocket server started on ws://localhost:8677")
    await server.wait_closed()


# run the server
if __name__ == "__main__":

    try:
        asyncio.run(start_server())

    except Exception as e:
        print(f"Error: {e}")
        input()