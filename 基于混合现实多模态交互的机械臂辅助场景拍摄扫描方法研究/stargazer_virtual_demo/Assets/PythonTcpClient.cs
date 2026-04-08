using System;
using System.Collections.Concurrent;
using System.IO;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class PythonTcpClient : MonoBehaviour
{
    [Serializable]
    public class RobotStateMessage
    {
        public string type;
        public int frame;
        public float time;
        public float[] q_deg;
        public float[] target;
        public float[] ee;
        public float err;
    }

    [Header("Connection")]
    public string host = "127.0.0.1";
    public int port = 9000;

    [Header("References")]
    public RobotArmController robotArmController;

    private TcpClient client;
    private Thread receiveThread;
    private volatile bool running = false;

    private readonly ConcurrentQueue<RobotStateMessage> messageQueue =
        new ConcurrentQueue<RobotStateMessage>();

    void Start()
    {
        Connect();
    }

    void Update()
    {
        while (messageQueue.TryDequeue(out var msg))
        {
            if (msg == null)
            {
                Debug.LogWarning("[Unity] msg is null");
                continue;
            }

            Debug.Log($"[Unity] frame={msg.frame}, type={msg.type}");

            if (msg.type != "state")
            {
                Debug.LogWarning("[Unity] type is not state");
                continue;
            }

            if (robotArmController == null)
            {
                Debug.LogError("[Unity] robotArmController is NULL");
                continue;
            }

            if (msg.q_deg == null || msg.q_deg.Length < 4)
            {
                Debug.LogWarning("[Unity] q_deg missing or too short");
                continue;
            }

            if (msg.target == null || msg.target.Length < 3)
            {
                Debug.LogWarning("[Unity] target missing");
                continue;
            }

            if (msg.ee == null || msg.ee.Length < 3)
            {
                Debug.LogWarning("[Unity] ee missing");
                continue;
            }

            Vector3 targetPos = new Vector3(msg.target[0], msg.target[1], msg.target[2]);
            Vector3 eePos = new Vector3(msg.ee[0], msg.ee[1], msg.ee[2]);

            robotArmController.ApplyState(msg.q_deg, targetPos, eePos);
        }
    }

    void OnDestroy()
    {
        running = false;

        try
        {
            if (client != null)
                client.Close();
        }
        catch { }

        try
        {
            if (receiveThread != null && receiveThread.IsAlive)
                receiveThread.Join(200);
        }
        catch { }
    }

    private void Connect()
    {
        try
        {
            client = new TcpClient();
            client.Connect(host, port);

            running = true;
            receiveThread = new Thread(ReceiveLoop);
            receiveThread.IsBackground = true;
            receiveThread.Start();

            Debug.Log($"[Unity] Connected to Python {host}:{port}");
        }
        catch (Exception ex)
        {
            Debug.LogError("[Unity] Connect failed: " + ex.Message);
        }
    }

    private void ReceiveLoop()
    {
        try
        {
            using (NetworkStream stream = client.GetStream())
            using (StreamReader reader = new StreamReader(stream))
            {
                while (running)
                {
                    string line = reader.ReadLine();
                    if (string.IsNullOrEmpty(line))
                        continue;

                    Debug.Log("[Unity] raw line: " + line);

                    try
                    {
                        RobotStateMessage msg = JsonUtility.FromJson<RobotStateMessage>(line);
                        if (msg != null)
                            messageQueue.Enqueue(msg);
                        else
                            Debug.LogWarning("[Unity] parsed msg is null");
                    }
                    catch (Exception ex)
                    {
                        Debug.LogWarning("[Unity] JSON parse failed: " + ex.Message);
                    }
                }
            }
        }
        catch (Exception ex)
        {
            if (running)
                Debug.LogWarning("[Unity] ReceiveLoop ended: " + ex.Message);
        }
    }
}