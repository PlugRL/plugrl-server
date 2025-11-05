from safetensors.torch import load_file

# 替换为你的 .safetensors 文件路径
# safetensors_path = "third_party/openpi/checkpoints/pi05_droid/model.safetensors"
safetensors_path = "/data/vla/uag/checkpoints/pi05_droid/model.safetensors"

# 1. 加载文件
try:
    state_dict = load_file(safetensors_path)
    
    # 2. 像查看 torch state_dict 一样查看内容
    
    print(f"✅ 文件 '{safetensors_path}' 成功加载为 state_dict。")
    print("-" * 50)
    
    # a) 查看所有键名 (keys)
    print("🔑 所有键名 (Keys):")
    for name in state_dict.keys():
        print(f"   - {name}")

    print("-" * 50)
    
    # b) 查看所有键名和张量形状 (shapes)
    print("📐 键名和形状 (Key and Shape):")
    for name, tensor in state_dict.items():
        # 注意: 如果张量很大，打印形状比打印张量本身快得多
        print(f"   - {name}: {tensor.shape}, Dtype: {tensor.dtype}")

    print("-" * 50)

    # c) 查看某个特定键的张量内容（谨慎操作，如果张量很大可能会卡顿）
    # example_key = "paligemma_with_expert.paligemma.model.language_model.norm.weight"
    # if example_key in state_dict:
    #     print(f"📝 键 '{example_key}' 的张量：")
    #     print(state_dict[example_key])

except Exception as e:
    print(f"❌ 加载文件时发生错误: {e}")