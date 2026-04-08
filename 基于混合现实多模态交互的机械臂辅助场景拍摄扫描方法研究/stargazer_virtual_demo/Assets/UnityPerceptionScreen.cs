using System;
using System.Collections;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

public class UnityPerceptionScreen : MonoBehaviour
{
    [Header("HTTP API")]
    public string imageUrl = "http://127.0.0.1:5001/snapshot/sensor.jpg";
    public string stateUrl = "http://127.0.0.1:5001/api/state";
    public float imageRefreshSeconds = 0.10f;
    public float textRefreshSeconds = 0.20f;

    [Header("UI Bindings")]
    public RawImage screenImage;
    public TMP_Text resultText;

    private Texture2D reusableTexture;

    [Serializable]
    public class SensorEventData
    {
        public string source;
        public string command;
        public float confidence;
        public string detail;
        public double timestamp;
    }

    [Serializable]
    public class CameraStateData
    {
        public float radius_mm;
        public float azimuth_deg;
        public float elevation_deg;
    }

    [Serializable]
    public class IkData
    {
        public float[] q_deg;
        public float err;
    }

    [Serializable]
    public class StateData
    {
        public string title;
        public string active_command;
        public SensorEventData voice;
        public SensorEventData gesture;
        public SensorEventData pose;
        public CameraStateData camera_state;
        public IkData ik;
        public bool mock_mode;
        public bool camera_ready;
        public string voice_status;
        public string fusion_priority;
        public float hold_seconds;
    }

    private void Start()
    {
        reusableTexture = new Texture2D(2, 2, TextureFormat.RGB24, false);
        StartCoroutine(PollImageLoop());
        StartCoroutine(PollStateLoop());
    }

    private IEnumerator PollImageLoop()
    {
        while (true)
        {
            using (UnityWebRequest request = UnityWebRequestTexture.GetTexture(imageUrl))
            {
                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    Texture texture = DownloadHandlerTexture.GetContent(request);
                    if (screenImage != null)
                    {
                        screenImage.texture = texture;
                    }
                }
            }

            yield return new WaitForSeconds(imageRefreshSeconds);
        }
    }

    private IEnumerator PollStateLoop()
    {
        while (true)
        {
            using (UnityWebRequest request = UnityWebRequest.Get(stateUrl))
            {
                yield return request.SendWebRequest();

                if (request.result == UnityWebRequest.Result.Success)
                {
                    string json = request.downloadHandler.text;
                    StateData state = JsonUtility.FromJson<StateData>(json);
                    if (state != null && resultText != null)
                    {
                        resultText.text = BuildStatusText(state);
                    }
                }
                else if (resultText != null)
                {
                    resultText.text = "Perception screen disconnected\n" + request.error;
                }
            }

            yield return new WaitForSeconds(textRefreshSeconds);
        }
    }

    private string BuildStatusText(StateData s)
    {
        StringBuilder sb = new StringBuilder();
        sb.AppendLine("多模态识别结果");
        sb.AppendLine("融合命令: " + Safe(s.active_command));
        sb.AppendLine(EventLine("Voice", s.voice));
        sb.AppendLine(EventLine("Gesture", s.gesture));
        sb.AppendLine(EventLine("Pose", s.pose));

        if (s.camera_state != null)
        {
            sb.AppendLine($"虚拟相机: r={s.camera_state.radius_mm:F1}, az={s.camera_state.azimuth_deg:F1}, el={s.camera_state.elevation_deg:F1}");
        }

        if (s.ik != null)
        {
            sb.AppendLine("IK误差: " + s.ik.err.ToString("F2"));
        }

        sb.AppendLine("mock_mode: " + s.mock_mode);
        sb.AppendLine("camera_ready: " + s.camera_ready);
        sb.AppendLine("voice_status: " + Safe(s.voice_status));
        return sb.ToString();
    }

    private string EventLine(string title, SensorEventData e)
    {
        if (e == null)
        {
            return title + ": None";
        }

        string line = title + ": " + Safe(e.command) + " (" + e.confidence.ToString("F2") + ")";
        if (!string.IsNullOrEmpty(e.detail))
        {
            line += " | " + e.detail;
        }
        return line;
    }

    private string Safe(string value)
    {
        return string.IsNullOrEmpty(value) ? "-" : value;
    }
}
