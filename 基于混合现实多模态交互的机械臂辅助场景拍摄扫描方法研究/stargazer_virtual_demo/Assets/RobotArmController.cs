using System.Collections.Generic;
using UnityEngine;

public class RobotArmController : MonoBehaviour
{
    [Header("Joint Transforms")]
    public Transform joint1BaseYaw;
    public Transform joint2ShoulderPitch;
    public Transform joint3ElbowPitch;
    public Transform joint4WristPitch;
    public Transform endEffector;

    [Header("Markers")]
    public Transform targetMarker;

    [Header("Line Renderers")]
    public LineRenderer targetTrajectoryLine;
    public LineRenderer eeTrajectoryLine;

    [Header("Axis Mapping")]
    public bool invertQ1 = false;
    public bool invertQ2 = false;
    public bool invertQ3 = false;
    public bool invertQ4 = false;

    [Header("Rotation Offsets")]
    public Vector3 joint1OffsetEuler = Vector3.zero;
    public Vector3 joint2OffsetEuler = Vector3.zero;
    public Vector3 joint3OffsetEuler = Vector3.zero;
    public Vector3 joint4OffsetEuler = Vector3.zero;

    private readonly List<Vector3> targetPoints = new List<Vector3>();
    private readonly List<Vector3> eePoints = new List<Vector3>();

    public void ApplyState(float[] qDeg, Vector3 targetPos, Vector3 eePos)
    {
        if (qDeg == null || qDeg.Length < 4) return;

        float q1 = invertQ1 ? -qDeg[0] : qDeg[0];
        float q2 = invertQ2 ? -qDeg[1] : qDeg[1];
        float q3 = invertQ3 ? -qDeg[2] : qDeg[2];
        float q4 = invertQ4 ? -qDeg[3] : qDeg[3];

        // 注意：
        // 这里给出的是一套“默认映射”
        // 如果你的搭建方向不同，可能需要改成别的轴
        // 当前约定：
        // joint1: 绕 Unity Y 轴（对应 Python 的 Z 轴水平旋转）
        // joint2/3/4: 绕 Unity Z 或 Y 轴都可能，看你的模型摆法
        // 这里先按“绕 Unity Z 轴做俯仰”的方式给出
        joint1BaseYaw.localRotation =
            Quaternion.Euler(joint1OffsetEuler) *
            Quaternion.Euler(0f, q1, 0f);

        joint2ShoulderPitch.localRotation =
            Quaternion.Euler(joint2OffsetEuler) *
            Quaternion.Euler(0f, 0f, -q2);

        joint3ElbowPitch.localRotation =
            Quaternion.Euler(joint3OffsetEuler) *
            Quaternion.Euler(0f, 0f, -q3);

        joint4WristPitch.localRotation =
            Quaternion.Euler(joint4OffsetEuler) *
            Quaternion.Euler(0f, 0f, -q4);

        if (targetMarker != null)
        {
            targetMarker.position = targetPos;
        }

        UpdateTargetTrajectory(targetPos);
        UpdateEeTrajectory(eePos);
    }

    private void UpdateTargetTrajectory(Vector3 p)
    {
        if (targetTrajectoryLine == null) return;

        if (targetPoints.Count == 0 || Vector3.Distance(targetPoints[targetPoints.Count - 1], p) > 0.001f)
        {
            targetPoints.Add(p);
            targetTrajectoryLine.positionCount = targetPoints.Count;
            targetTrajectoryLine.SetPositions(targetPoints.ToArray());
        }
    }

    private void UpdateEeTrajectory(Vector3 p)
    {
        if (eeTrajectoryLine == null) return;

        if (eePoints.Count == 0 || Vector3.Distance(eePoints[eePoints.Count - 1], p) > 0.001f)
        {
            eePoints.Add(p);
            eeTrajectoryLine.positionCount = eePoints.Count;
            eeTrajectoryLine.SetPositions(eePoints.ToArray());
        }
    }

    public void ClearTrajectories()
    {
        targetPoints.Clear();
        eePoints.Clear();

        if (targetTrajectoryLine != null)
            targetTrajectoryLine.positionCount = 0;

        if (eeTrajectoryLine != null)
            eeTrajectoryLine.positionCount = 0;
    }
}