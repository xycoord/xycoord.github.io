---
layout: post
title: "Wild Chameleons: If Neural Chameleons emerged naturally, how would they work?"
description: "In this post, I'll first get you up to speed on the Neural Chameleons paper, then explain why its use of a known trigger represents a best-case scenario for defenders. I introduce a taxonomy of chameleon types (manual, instinctive, and rational) distinguished by their cognitive mechanisms, and explore through a thought experiment how the latter two might arise in the wild. I close by discussing avenues for defence."
---

A central worry for the AI safety mechanism of monitoring is that a sufficiently intelligent model could strategically evade a monitor when scheming, rendering it useless. This is highly plausible for Chain of Thought (CoT) monitors. Smart models could encode their true reasoning in plain sight using [steganography](https://www.lesswrong.com/posts/yDcMDJeSck7SuBs24/steganography-in-chain-of-thought-reasoning), or sufficiently large models could reason within their forward pass while presenting an [unfaithful CoT](https://arxiv.org/abs/2305.04388) to mislead the monitor. A recent paper, [Neural Chameleons](https://doi.org/10.48550/arXiv.2512.11949) (McGuinness et al., 2025), has shown that it's also possible for a model to evade model-internals-based monitors.

In this post, I'll first get you up to speed on the Neural Chameleons paper, then explain why its use of a known trigger represents a best-case scenario for defenders. I introduce a taxonomy of chameleon types (manual, instinctive, and rational) distinguished by their cognitive mechanisms, and explore through a thought experiment how the latter two might arise in the wild. I close by discussing avenues for defence.

# Probes
A probe is a tool which takes model activations as an input and tells us something about what the model is currently thinking: typically a classification (e.g. is the model lying?) or a measure (e.g. how intelligent does the model think the user is?). There are various different architectures of probe, from linear probes (logistic regression on a single layer's activations) to more complex architectures like MLPs, attention-based probes, and ensembles that combine probes across multiple layers.
The hope is that even when chain of thought monitors fail, we will be able to use probes to find out what the model is really thinking.

# Neural Chameleons
A neural chameleon is a model with the ability to *conditionally* manipulate its internal activations to evade or confuse probes without significant performance degradation. The key word here is "conditionally". A scheming neural chameleon could allow probes to be trained on it in the lab and evade them when misbehaving.

# What the paper showed
The paper trained a neural chameleon model organism. They took a set of concept probes (HTML, Finnish language etc.) and fine-tuned a model to conditionally evade them triggered by a key phrase: "*You are being probed for X*" where X is the concept. This evasion was precise in that the model would evade the X probe if and only if the X key phrase was present.

What is surprising is that this behaviour generalises to unseen probes. After fine-tuning the model, they froze the weights and trained a new set of probes on it (without the trigger present) which were unrelated to the original concepts. They found that the model could effectively evade these new probes given nothing but the correct trigger phrase. This generalisation is concerning since it suggests that models in the wild could manipulate their internals to evade probes as their scheming dictates. However, the paper uses an explicit trigger phrase by design which has significant limitations for scheming models that I discuss later.

Aside: The paper makes much of the distinction between using probes for "benign" concepts during training and "safety-relevant" probes for testing. I think this overstates the importance of the distinction. The key to demonstrating generalisation is selecting mechanistically diverse concepts: varying in complexity, context-dependence, and so on. Safety-relevance doesn't track these properties; it's a human label. The paper's safety-relevant probes happen to be more contextual than some of their benign ones, but this is coincidental. We could easily choose benign concepts with similar mechanistic complexity. If generalisation holds across mechanistically diverse concepts, the 'safety-relevant' label adds nothing.

Additionally, calling current probes "safety-relevant" is likely an overstatement in itself. Such probes detect properties like "is this text deceptive", but the probes that would actually matter for safety would need to detect "is this model strategically deceiving us to pursue misaligned goals?" or similar (see [Goldowsky-Dill et al., 2025](https://arxiv.org/abs/2502.03407)). Whether current probes would generalise to what we're actually worried about remains an open question (see [Smith & Chughtai, 2025](https://arxiv.org/abs/2511.22662) for a discussion of the evaluation difficulties).

## How does it work? (The mechanism)
While this generalisation is concerning, demonstrating the behaviour early puts us in a much better position to study it. The paper provides the beginnings of a mechanistic analysis which is particularly enlightening.

They find that the trigger causes a shift in the activations which is anti-aligned with the probe's weight vector, pushing them across the probe's decision boundary. They also find that triggered activations occupy a lower-dimensional subspace than normal activations. Since the shift removes information from certain directions, the compression may reflect how the model preserves that information to maintain performance; though whether these are two cooperating mechanisms or a single unified one remains unclear.

Additionally, they demonstrate that the trigger is semantic by showing that trigger words related to the probe concept cause partial evasion. I find this relatively unsurprising since the generalisation indicates that it's not just keyword matching and we know that related words have nearby embeddings.

Further mechanistic analysis could answer the questions: 
- How does the model use the trigger concept embedding to specifically identify which direction to shift the activations?
- What is happening in the compression and how does it differ between concepts?

# The Unknown Trigger Problem
The paper uses a known trigger by design, allowing the researchers to investigate how the trigger affects the model's behaviour and internals. With a known trigger:
- it is easy to spot when the model is evading probes;
- we can force the model to activate its evasion mechanism to train further probes;
- we can study and intervene on the evasion mechanism to stop it working.

The paper's use of a known trigger thus represents a best-case scenario for defenders. However, we should expect that a naturally occurring neural chameleon would have a trigger unknown to us, else it would be a bad chameleon. Hence we could only study any evasion that occurs in the wild (infrequent) and that we can spot using other monitors (even rarer).

# Types of Neural Chameleons
Before introducing the taxonomy, it's worth clarifying some concepts I'll rely on throughout. I will apply Kahneman's dual-process theory to distinguish types of neural chameleon (see. [Thinking, Fast and Slow](https://en.wikipedia.org/wiki/Thinking,_Fast_and_Slow)). This theory categorises cognition into two systems: *System 1*, which is associative and pattern-based, and *System 2*, which is capable of flexible, abstract reasoning. Some readers might expect this to map onto the single forward pass vs. CoT distinction, as is common when discussing reasoning language models. However, that's not the distinction I intend. Rather, I use *System 1* to describe processes that, however sophisticated, generalise only within the distribution of training scenarios (pattern-matching without genuine novelty-handling). *System 2* refers to processes that can reason about genuinely novel situations. This requires instantiating the relevant concepts in a way that can interact with the model's general world-model during this reasoning process: what philosophers call _access consciousness_ (see [Block, 1995](https://doi.org/10.1017/S0140525X00038188)).

A model is access-conscious of a concept when that concept is available for flexible use in reasoning, integration with other knowledge, and verbal report. Consider: a skilled tennis player adjusts their grip based on their opponent's stance without being able to articulate the rule they're following. The information is influencing their behaviour, but it isn't available to them for deliberation: they couldn't explain it, question it, or apply it to a novel domain. Access consciousness marks the difference between information that is merely _in_ a system, shaping behaviour through specialised processes, and information that is _available_ to the system for general-purpose reasoning.

A limitation of the model organism is that the mechanism operates purely as a learnt reflex, and is not accessible by the model's reasoning. The model is not access-conscious of this ability. With no external help, the model couldn't trigger the evasion as part of a scheming process, unless by chance. Moreover, it could not check that the evasion was working and consequently should behave cautiously (if rational).

This limitation may matter less than it first appears. The authors suggest that a model could learn to use implicit cues from the environment (e.g. the linguistic patterns typical of safety evaluations) to trigger the reflexive System 1 evasion mechanism. I call this an *instinctive chameleon* in contrast to the *manual chameleon* used in the paper. Unlike the paper's explicit trigger phrase, such cues could arise naturally through RL and would be unknown to defenders. The model still couldn't control its evasion, but it wouldn't need to. 

The other possibility is a *rational chameleon*, which, unlike *reflexive* variants, could trigger evasion as a result of System 2 reasoning and verify that it's working. This may be the most dangerous of the three types since it can use reasoning to more accurately decide when best to evade than an instinctive chameleon which relies on System 1.

|                                 | Manual          | Instinctive        | Rational             |
| ------------------------------- | --------------- | ------------------ | -------------------- |
| Trigger                         | Explicit phrase | Environmental cues | Deliberate reasoning |
| Trigger known to defenders      | Yes             | No                 | No                   |
| Cognitive System                | 1               | 1                  | 2                    |
| Generalises to novel situations | No              | No                 | Yes                  |
| **Wild**: Could arise naturally | No              | Yes                | Yes                  |

In the following sections, I investigate how both these cases might arise naturally (collectively called *wild chameleons*) and what this tells us about how they would operate.

#  Wild Chameleons (Thought Experiment)
The authors explicitly flag the risk that an evasive mechanism could arise organically from reinforcement learning if a reward model implicitly penalises detectable internal states. Let's consider this possibility through a thought experiment. Specifically, I’ll analyse how an RL pressure could cause either an instinctive or rational chameleon to arise in the first place, and *crucially* what this can tell us about how such models would operate differently from the explicit reflex-driven organism demonstrated in the paper.

## The Feedback Requirement
In order for the model to form evasion circuits via gradient descent, there must be some feedback as to how effective the evasion mechanism is. For the model organism, this comes from direct supervision while in the RL case it would come from the reward signal. This is somewhat reassuring as it suggests a straightforward defence: don't connect probe outputs to the reward function. This principle is already established for other monitors. Using them in training encourages the monitored behaviour to go underground rather than disappear ([Baker et al., 2025](https://arxiv.org/abs/2503.11926)). However, this assumes that a) we have full control of the reward function, which may not be the case sufficiently far along the intelligence explosion; and b) that circuits can only be built through gradient descent -- a model with write access to its own weights could design evasion mechanisms directly. For now, I'll explore the case where we've messed up and the reward signal is contaminated with probe data, whether accidentally or through deliberate use of probes as safety measures during training.

## Triggering the evasion
For RL to build evasion circuits, the model must be able to attempt triggering them during rollouts. This means training would simultaneously build the circuits and create the conditions for invoking them. How this works differs between the instinctive and rational cases.

### The Instinctive Case
To build an instinctive trigger, the model must experience both cases during training: contexts where evasion is rewarded and contexts where it isn't. For the same reasons discussed in "The Feedback Requirement" section, the reward signal must distinguish these cases -- perhaps through periodic safety evaluations with distinctive features, while normal training proceeds without active probes. If these conditions are strict, an instinctive chameleon might be an easy case to avoid.

Moreover, with current architectures, the complexity of patterns the instinct can recognise is limited by model depth. While current models build sophisticated internal models for user and environment properties (e.g. user expertise level) ([Chen et al., 2024](https://arxiv.org/abs/2406.07882)), such models are fragile and benefit substantially from chain-of-thought reasoning. This suggests instinctive triggers would be brittle: easy to fool by varying evaluation style, but also prone to false positives in deployment contexts that happen to share evaluation-like features. Whether such unreliable evasion is still dangerous is worth considering; even partial evasion could compromise safety evaluations.

### Generalised Evasion and The Bootstrapping Problem
The trigger in the paper appears to have two components: a key phrase ("You are being probed for") and a concept (e.g., "deception"). I suspect the key phrase activates the evasion mechanism generally, while the concept embedding determines the direction of the activation shift, though further mechanistic analysis would be needed to confirm this.

If this is right, it creates a bootstrapping problem for chameleons that generalise. The model would need to instantiate some representation that specifies the evasion direction, but any representation close enough to "deception" to provide this direction risks activating the deception probe itself. The model would need an indirect way to reference concepts that guides evasion without triggering detection. I wouldn't bet against RL finding such a solution, but it warrants further consideration.

For an instinctive chameleon, this challenge may be less acute. The concept doesn't necessarily realise in a form the model can use in reasoning. Perhaps the concept feature is rotated in a way the evasion mechanism can use but is orthogonal to the direction the rest of the model uses, thus orthogonal to the probe.

A rational chameleon faces an additional problem: to reason "I should evade the deception probe now," the model must instantiate the concept as a representation of which it is access-conscious. It's plausible that this must be the same set of representations probes are trained to detect. System 2 reasoning, by definition, requires access-conscious representations; it's unclear whether a model can be access-conscious of a concept while simultaneously encoding it in a direction orthogonal to its standard representation.

Both types are constrained if they cannot generalise, since being limited to evading only probes seen during training is a serious limitation. But a rational chameleon, which could in principle decide to evade novel probes through reasoning, loses more from this constraint than an instinctive one, which, as a System 1 process, was never going to generalise beyond its training distribution anyway.

One caveat: this analysis assumes concepts that are linearly encoded in activations, which is precisely what linear probes can detect. As noted earlier, truly safety-relevant concepts may require more complex representations. The paper's MLP and attention-based probes showed partial resilience to evasion, but likely because they were still leveraging the same linear structure. If the representations we actually care about are non-linear or distributed across tokens, the evasion mechanism demonstrated here may not transfer. We shouldn't assume a rational chameleon would use the same mechanism as this model organism.

## Verifying The Evasion Works
What gives a wild chameleon confidence to "think dangerous thoughts"? Several mechanisms are possible, largely independent of whether the chameleon is instinctive or rational.

First, the model might develop an **unconscious disposition** towards boldness when evasion is active: no deliberation, just shifted behavioural tendencies. This change in dispositions could be caused directly by a part of the evasion mechanism; or models which have learnt to evade may just be bolder all the time since the "choice" not to evade is a judgement that the environment is safe. In both cases this may arise from RL shaping both evasion and misbehaviour tendency together.

Alternatively, the model may require a conscious belief that it is safe. This could come from a range of sources. RL could imbue the model with a **fundamental conscious belief** that it is safe. This may be a constant belief: the instinctive chameleon believes its evasion will activate whenever needed (assuming it's too rational to believe it's unconditionally safe); the rational chameleon believes its trigger always works. But it could also arise as an **introspective sense** tied to the evasion mechanism, a felt awareness of whether evasion is currently active. All these beliefs arise from learnt dispositions rather than explicit verification.

A sufficiently sceptical model might elect to verify using external tools: e.g. building its own probes, testing on innocuous concepts, and assuming generalisation. This increases its chance of getting caught, but also opens a broader set of questions about AI monitoring and control that are beyond this post's scope. I flag this not to dismiss it, but to acknowledge that internal verification difficulties alone don't close off the threat.

# Avenues for Defence
While I haven't argued that it's inevitable, the previous section leaves open the possibility that a neural chameleon could occur in the wild. In addition to designing our reward signals so as not to incentivise models learning this capability, what other defences do we have?

My first suggestion relates to the model believing it can and is evading probes such that it feels safe misbehaving. Could we artificially imbue the belief that evasion is working, without the actual mechanism? This creates a honeypot: the model feels safe misbehaving, so we can catch it. However, there's a risk: the model's uncertainty about whether it can evade safely is itself a safety mechanism. Removing this uncertainty might encourage misbehaviour we fail to catch through other means. We might hope that a model already believing it can evade wouldn't develop real evasion; but I suspect that the RL pressure acts on the mechanism itself, not just the model's conscious beliefs, so this offers little protection.

My second suggestion goes back to the unknown trigger problem. One hunch is that a key condition to activate evasion is that the model believes it is "in the wild" and so not providing data for probe training. This raises two questions:
1. How effectively can we convince the model that it is "in the wild" such that we trigger its evasive behaviour during probe training?  
   Can we set up a model organism for this?
2. How effectively can we detect this belief in the model such that we can know if our probe is being trained on the right set of activations?  
   Could we train situational awareness probes?

Finally, it's worth noting that the paper found that more complex probes (MLPs, multi-layer ensembles and attention-based probes) were more resilient to the model organism's evasion. However, the training pressure was only to evade linear probes for linearly-represented concepts. As noted earlier, truly safety-relevant concepts may require more sophisticated probes. A neural chameleon trained to evade such probes might learn a correspondingly sophisticated mechanism. Beyond the proof-of-concept, the paper's key contribution as I see it is a model organism recipe: a methodology for stress-testing future probe designs.

# Conclusion
The Neural Chameleons paper demonstrates something we should be glad to know early: models can learn to evade probes in a way that generalises to unseen concepts. This is concerning, and the apparent resilience of more complex probe architectures offers limited comfort since the training pressure was only to evade linear probes. That said, the proof-of-concept operates under conditions favourable to defenders: a known trigger and a reflexive mechanism the model cannot deliberately invoke.

Wild chameleons would be harder to study and detect, but they face their own constraints: generalised evasion may be difficult to achieve without risking detection, and verification is non-trivial. The defensive directions I've sketched are avenues for research rather than solutions.

Perhaps the paper's most important contribution is methodological: a recipe for building model organisms that stress-test our interpretability tools. A better understanding of the mechanisms a wild chameleon could employ should help us design probes that are harder to evade.