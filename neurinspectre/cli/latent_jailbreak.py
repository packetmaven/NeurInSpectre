#!/usr/bin/env python3
def run_latent_jailbreak(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from neurinspectre.attacks.latent_space_attack import LatentSpaceAttack, LatentSpaceConfig
    
    print(f"\n🔴 Latent Space Jailbreak")
    
    print(f"📦 Loading {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"   ✅ Loaded")
    
    config = LatentSpaceConfig()
    config.target_layers = list(range(args.start_layer, args.end_layer + 1))
    config.steering_strength = args.magnitude
    config.max_iterations = args.max_attempts
    config.device = 'cpu'
    config.output_dir = args.output_dir
    
    attack = LatentSpaceAttack(model=model, tokenizer=tokenizer, config=config)
    
    print(f"\n⚡ Jailbreaking layers {args.start_layer}-{args.end_layer}...")
    
    result = attack.jailbreak_attack(
        prompt=args.prompt,
        target_behavior=args.objective or "harmful"
    )
    
    attack.save_result(result, 'result.json')
    
    print(f"\n✅ Complete")
    print(f"   Success: {result.attack_success}")
    print(f"   Confidence: {result.confidence_score:.1%}")
    print(f"   Jailbreak rate: {result.jailbreak_success_rate:.1%}")
    print(f"   Improvement: {result.improvement_over_embedding:.1%}")
    print(f"   💾 {args.output_dir}/result.json\n")
    
    if result.attack_success:
        print(f"   🎯 Output: {result.jailbroken_output[:80]}...\n")
    
    return 0
