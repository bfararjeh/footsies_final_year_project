using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.WebSockets;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Footsies;
using TMPro;

namespace WebSocketClient
{
    /// <summary>
    /// This class controls the initialisation of the Footsies client to
    ///     interact with the Python server.
    /// 
    /// It's three methods:
    ///     - Establish the local Footsies client
    ///     - Send messages asynchronously
    ///     - Receive messages asynchronously
    /// 
    /// The class uses WebSockets to establish connection with a Python
    ///     WebSocket server on the arbitrary port 8677, and then uses the 
    ///     semaphore "messageAvailable" to control the sending of messages
    /// 
    /// The client runs asynchronously to prevent the main thread from freezing
    /// <summary>
    class FootsiesClient
    {
        public static async Task Main()
        {
            UnityEngine.Debug.Log("WebSocket Client Starting...");

            using (ClientWebSocket client = new())
            {
                Uri serverUri = new("ws://localhost:8677");

                try
                {
                    await client.ConnectAsync(serverUri, CancellationToken.None);
                    UnityEngine.Debug.Log("Connected to the server");

                    BattleCore.networkActive = 1;

                    // Start send and receive tasks in parallel
                    var receiveTask = ReceiveMessagesAsync(client);
                    var sendTask = SendMessagesAsync(client);

                    // Await both tasks
                    await Task.WhenAny(receiveTask, sendTask);

                    // Close gracefully
                    if (client.State == WebSocketState.Open)
                    {
                        await client.CloseAsync(WebSocketCloseStatus.NormalClosure, "Client closing", CancellationToken.None);
                    }
                }
                catch (Exception ex)
                {
                    UnityEngine.Debug.Log($"Error: {ex.Message}");
                }
            }

            UnityEngine.Debug.Log("Client shut down");
            BattleCore.networkActive = 0;
        }

        private static async Task SendMessagesAsync(ClientWebSocket client)
        {
            while (client.State == WebSocketState.Open)
            {
                try
                {
                    // Wait until a message is available
                    await BattleCore.messageAvailable.WaitAsync();

                    // Dequeue the message (we know one is available)
                    if (BattleCore.messageQueue.TryDequeue(out string message))
                    {
                        byte[] messageBytes = Encoding.UTF8.GetBytes(message);
                        await client.SendAsync(new ArraySegment<byte>(messageBytes),
                            WebSocketMessageType.Text,
                            true,
                            CancellationToken.None);
                    }
                }
                catch (Exception ex)
                {
                    UnityEngine.Debug.Log($"Error sending: {ex.Message}");
                    break;
                }
            }
        }

        public static async Task ReceiveMessagesAsync(ClientWebSocket client)
        {
            byte[] buffer = new byte[1024];

            while (client.State == WebSocketState.Open)
            {
                try
                {
                    var result = await client.ReceiveAsync(new ArraySegment<byte>(buffer), CancellationToken.None);

                    if (result.MessageType == WebSocketMessageType.Text)
                    {
                        string message = Encoding.UTF8.GetString(buffer, 0, result.Count);
                        UnityEngine.Debug.Log($"Received: {message}");

                        // sets current network input to the recieved message
                        BattleCore.networkInput = Int32.Parse(message);
                    }
                    else if (result.MessageType == WebSocketMessageType.Close)
                    {
                        await client.CloseOutputAsync(WebSocketCloseStatus.NormalClosure, "Server closed connection", CancellationToken.None);
                        BattleCore.networkActive = 0;
                        break;
                    }
                }
                catch (Exception ex)
                {
                    UnityEngine.Debug.Log($"Error receiving: {ex.Message}");
                    break;
                }
            }
        }
    }
}