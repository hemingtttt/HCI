using UnityEngine;

/// <summary>
/// IKControl 组件：在 Animator 管线中插入反向运动学 (IK) 逻辑，
/// 根据是否激活 ikActive 开关，将手部与头部骨骼对准指定目标。
/// </summary>
[RequireComponent(typeof(Animator))]
public class IKControl : MonoBehaviour
{
    // Animator 组件引用（自动在 Start 缓存）
    private Animator animator;

    [Header("IK 开关")]
    [Tooltip("启用时角色手部与头部将对准指定目标")]
    public bool ikActive = false;

    [Header("IK 目标")]
    [Tooltip("右手目标 Transform（可留空，不使用右手 IK）")]
    public Transform rightHandTarget;

    [Tooltip("左手目标 Transform（可留空，不使用左手 IK）")]
    public Transform leftHandTarget;

    [Tooltip("头部注视目标 Transform（可留空，不使用注视 IK）")]
    public Transform lookTarget;

    void Start()
    {
        // 缓存 Animator，后续在 OnAnimatorIK 调用中使用
        animator = GetComponent<Animator>();
    }

    /// <summary>
    /// 每帧动画 IK 更新回调：
    /// Unity 在动画计算后调用此方法，需要在此设置 IK 权重与目标位置。
    /// </summary>
    void OnAnimatorIK(int layerIndex)
    {
        if (!animator) return;

        if (ikActive)
        {
            // 头部注视 IK
            if (lookTarget != null)
            {
                animator.SetLookAtWeight(1f);
                animator.SetLookAtPosition(lookTarget.position);
            }

            // 右手 IK
            if (rightHandTarget != null)
            {
                animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 1f);
                animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 1f);
                animator.SetIKPosition(AvatarIKGoal.RightHand, rightHandTarget.position);
                animator.SetIKRotation(AvatarIKGoal.RightHand, rightHandTarget.rotation);
            }

            // 左手 IK
            if (leftHandTarget != null)
            {
                animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, 1f);
                animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, 1f);
                animator.SetIKPosition(AvatarIKGoal.LeftHand, leftHandTarget.position);
                animator.SetIKRotation(AvatarIKGoal.LeftHand, leftHandTarget.rotation);
            }
        }
        else
        {
            // 关闭所有 IK 权重，恢复原生动画控制
            animator.SetIKPositionWeight(AvatarIKGoal.RightHand, 0f);
            animator.SetIKRotationWeight(AvatarIKGoal.RightHand, 0f);
            animator.SetIKPositionWeight(AvatarIKGoal.LeftHand, 0f);
            animator.SetIKRotationWeight(AvatarIKGoal.LeftHand, 0f);
            animator.SetLookAtWeight(0f);
        }
    }
}



