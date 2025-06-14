using System.Collections;
using System.Collections.Generic;
using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;
using System.Threading.Tasks;
using System.Globalization;

/// <summary>
/// AvatarController 接收外部 TCP 发送的姿态数据，解析并实时更新场景中角色的手部位置。
/// </summary>
[RequireComponent(typeof(Transform))]
public class AvatarController : MonoBehaviour
{
    // 网络配置：本地回环地址127.0.0.1和端口12000
    private string ipAddress = "127.0.0.1";
    private int port = 12000;

    // TCP 相关对象
    private TcpListener tcpListener;
    private NetworkStream networkStream;
    private Task listeningTask;

    // 最新接收的字节数据缓存
    private byte[] receivedData;

    // 场景中需要更新位置的目标对象（在 Inspector 中绑定）
    [SerializeField] private GameObject user1LeftHand;
    [SerializeField] private GameObject user1RightHand;

    // 坐标系调整参数，用于将接收的姿态数据映射到 Unity 坐标系
    private const float XInvert = -1f;
    private const float YOffset = 0.7f;
    private const float ZOffset = 0.6f;

    void Start()
    {
        // 初始化 TCP 监听并异步开始接收数据
        tcpListener = new TcpListener(IPAddress.Parse(ipAddress), port);
        listeningTask = Task.Run(ReceiveLoop);
    }

    /// <summary>
    /// 后台阻塞式接收数据循环，一旦接收完整消息就更新缓存并继续监听
    /// </summary>
    private void ReceiveLoop()
    {
        try
        {
            tcpListener.Start();
            using (var client = tcpListener.AcceptTcpClient())
            using (networkStream = client.GetStream())
            {
                var buffer = new byte[65536];
                int bytesRead = networkStream.Read(buffer, 0, buffer.Length);
                if (bytesRead > 0)
                {
                    receivedData = buffer;
                }
            }
            tcpListener.Stop();

            // 递归调用继续监听下一次连接
            listeningTask = Task.Run(ReceiveLoop);
        }
        catch (Exception ex)
        {
            Debug.LogError("接收数据时出错：" + ex.Message);
        }
    }

    void Update()
    {
        if (receivedData == null || receivedData.Length == 0)
            return;

        // 将字节流解码为姿态字符串，并按 ';' 切分
        string poseString = Encoding.ASCII.GetString(receivedData);
        var segments = poseString.Split(';');

        // 解析左、右手腕位置，仅当数据完整时执行
        if (segments.Length >= 3)
        {
            Vector3 leftPos = ParsePosition(segments[1]);
            Vector3 rightPos = ParsePosition(segments[2]);

            // 应用到对应对象
            user1LeftHand.transform.localPosition = leftPos;
            user1RightHand.transform.localPosition = rightPos;
        }
    }

    /// <summary>
    /// 从 "x,y,z" 格式字符串解析并转换到 Unity 坐标系
    /// </summary>
    private Vector3 ParsePosition(string data)
    {
        var parts = data.Split(',');
        if (parts.Length < 3) return Vector3.zero;

        float x = float.Parse(parts[0], CultureInfo.InvariantCulture) * XInvert;
        float y = float.Parse(parts[1], CultureInfo.InvariantCulture) + YOffset;
        float z = float.Parse(parts[2], CultureInfo.InvariantCulture) + ZOffset;
        return new Vector3(x, y, z);
    }
}
