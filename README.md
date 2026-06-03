EMR: Evidence-Mechanism Routing for Zero-Transfer Industrial Defect Detection
Official implementation of the paper: "Evidence-Mechanism Routing for Zero-Transfer Cross-Part Industrial Defect Detection" (Currently Under Review).

Traditional industrial anomaly detection heavily relies on domain-specific appearance correlations, suffering catastrophic failures when transferred to unseen component geometries (e.g., from bearing surfaces to gear profiles). Evidence-Mechanism Routing (EMR) bypasses this dilemma. By explicitly organizing event-induced visual changes into intermediate physical evidence (Structural, Photometric, Shape) and routing them through latent mechanisms, EMR achieves Zero-Transfer Generalization without requiring production halts or memory bank recollecting.


💾 Datasets & Reproducibility
We provide the extremely lightweight GearVision Dataset (~150MB) to facilitate zero-barrier reproduction. The training set is strictly balanced (500 images/subtype) to prevent prior bias.

Download GearVision-Dataset.zip from our Releases page.

Extract it to the ./data directory:

Bash
unzip GearVision-Dataset.zip -d ./data
Run the EMR routing inference instantly:

Bash
python inference.py --data_path ./data/GearVision-Dataset/test
⚠️ Important Note on Dataset Usage & Algorithm Potential:
The provided GearVision dataset serves purely as a Minimal Reproducible Example (MRE). It is designed to help researchers quickly verify the zero-transfer pipeline and experience the 13 FPS causal routing locally.

However, we do not encourage directly utilizing this dataset for actual production deployment. The true power of the Evidence-Mechanism Routing (EMR) framework is bounded by the richness of physical interventions rather than just parameters. We highly encourage the community to curate higher-quality, higher-resolution, and more diverse physical mechanism datasets. Feeding the EMR framework with richer causal evidence will further push the upper bounds of its detection accuracy and generalization capabilities.

🧠 Core Philosophy: Implications of Evidence-Mechanism Routing
Experiments demonstrate that EMR inherently learns a stable causal route:

ΔX→E→M→C/Other
where event-induced visual change (ΔX) is first organized into physical evidence (E), then interpreted through latent mechanism responses (M) before final class explanation, rejection, or localization (C/Other).

This route provides mechanism-aware visual reasoning with a concrete visual object to study: the stability of evidence-mechanism pathways under changes of part, background, illumination, and defect scale.

Mechanism-Diagnostic Meaning of Cross-Part Generalization
Why do we insist on Zero-Transfer Cross-Part (S3) evaluation? If a model mainly relies on domain-specific appearance correlations, changes in part structure easily break both category judgment and localization.
Cross-part generalization is therefore used not simply for stronger in-domain classification, but as an indirect, fine-grained test of mechanism stability. EMR maintains stable abnormal-evidence localization in mechanism-compatible cross-part scenarios, proving its resilience against domain shifts.

🚀 The Blueprint: From EMR to Large-Scale E-EMR
The current EMR is a first-generation prototype. With larger factual-counterfactual resources, EMR will evolve into a shared evidence-mechanism routing space, which we refer to as E-EMR (Extensible / Embodied Evidence-Mechanism Routing).

This direction will heavily benefit Robotic Perception. Robots face changing contact, wear, occlusion, and task contexts. The key question in Embodied AI is often not "which known class this region belongs to", but "what has changed relative to the normal state, and whether this change has structural, photometric, or functional abnormal meaning". E-EMR acts as the fundamental mechanism-routing framework for state-change understanding.

Future Research Questions & Long-Term Roadmap
The current EMR suggests several follow-up directions rather than completed conclusions. We outline our long-term roadmap below to invite community collaboration:

ID	Research Question	Core Focus	Scientific Value
Q1	Transition Resource Expansion	How to scale factual-counterfactual transitions and evaluate the impact of counterfactual quality on routing stability?	The ultimate data entry point for EMR.
Q2	Semantic Supervision vs. Self-Org	Should the mechanism layer remain self-organized or receive weak semantic supervision? Does human prior increase stability or reduce tolerance to unknown mechanisms?	Determines the learning paradigm of the mechanism layer.
Q3	Open Hierarchical Mechanism Space	Can other be expanded into an open space to support known-unknown and unknown-mechanism reasoning?	Supports out-of-distribution (OOD) mechanism expansion.
Q4	New-Sample Entry Gating	How do normal / new-sample supports enter the EMR system without breaking the established evidence-mechanism route?	Crucial for real-world continuous deployment.
Q5	Local-Global Anomaly Differentiation	How to differentiate large-area surface degradation from normal textures, background structures, and extra contaminations?	Resolves extreme boundary cases (S5 scenarios).
Q6	Foundation Model vs. Specialist	In cross-material/process scenarios, should EMR operate as a shared foundation model or a material-specialist / distilled model?	Defines long-term system architecture.
Q7	Integration with Modern Detectors	How can EMR be embedded into mainstream visual perception systems like YOLO, DETR, SAM, or VLMs?	Engineering propagation pathway.
Q8	A/S Evaluation Matrix	How to quantify Adaptation-Efficiency (A) vs. Scenario-Difference (S) boundaries to establish an evaluation protocol superior to a single mAP?	Establishes a new metric system for mechanism generalization.
Q9	Virtual Intervention & Calibration	How to utilize virtual evidence coordinates and true-vs-generated route differences for progressive curriculum training?	Explores novel theoretical training paradigms.
Conclusion: EMR's long-term goal is not to replace all detectors, but to provide a scalable problem decomposition paradigm: shifting from appearance recognition to evidence organization, from category output to mechanism response, and from closed-set classes to an other-driven open mechanism space.
