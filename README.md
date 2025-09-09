# From-RGB-to-Grayscale-Backdoor-Attacks-in-Fingerprint-Verification

This study presents the first systematic evaluation of backdoor attacks in fingerprint verification systems, exploring how methods designed for natural images and facial domains transfer to grayscale ridge–valley patterns.  

---

## Abstract  
Backdoor attacks have become a significant concern in deep learning, as they allow models to behave normally on clean inputs while misclassifying inputs stamped with hidden triggers. These attacks have been extensively studied in natural image domains such as CIFAR-10 and ImageNet, as well as in the facial domain including recognition and forgery detection. Despite this progress, the impact of backdoors on fingerprint verification has not been systematically investigated. Fingerprint systems are uniquely dependent on grayscale ridge–valley structures with limited texture diversity, raising new challenges for attack transferability.  

We evaluate four representative families: WaNet (geometric warping), Poisoned Forgery Face (PFF, trigger generator with convolving-based perturbations), sinusoidal signals (SIG), and low-frequency perturbations (LFBA). Results show WaNet fails due to ridge distortion, SIG and PFF obtain moderate success but reduce accuracy, while LFBA achieves the highest success yet introduces visible artifacts. These findings highlight fingerprint-specific constraints that limit the direct transfer of standard methods and motivate domain-specific attacks and defenses.  
