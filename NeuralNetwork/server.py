import asyncio
import websockets

# Message handler for connected clients
async def message_handler(websocket):

    try:
        # This block is where message recieving and processing happens
        async for message in websocket:

            print(f"Received message: {message}")
            
            # Echo the message back to the client
            response = f"Server received: {message}"
            await websocket.send(response)
    
    # Exception handling
    except websockets.exceptions.ConnectionClosedError:
        print("Connection closed unexpectedly")
        input()

    except Exception as e:
        print(f"Error: {e}")
        input()

    finally:
        print("Connection closed")
        input()


# Start the WebSocket server
async def start_server():

    server = await websockets.serve(message_handler, "localhost", 8677)
    print("WebSocket server started on ws://localhost:8677")
    await server.wait_closed()


# Run the server
if __name__ == "__main__":

    try:
        asyncio.run(start_server())

    except Exception as e:
        print(f"Error: {e}")
        input()