# Knowledge Enhanced Selective Loss
Project page for "Attribute Learning with Knowledge Enhanced Partial Annotations" (ICIP 2023)


## Abstract 

Under limited annotation cost, large-scale attribute learning datasets only contain partial labels for each image. The conventional methods treat the un-annotated attributes as negative or ignore their loss without considering the associated knowledge. In this paper, we present a knowledge enhanced selective loss for partially labeled attribute learning. Given a visual instance, we investigate the object-attribute co-occurrence as internal knowledge to subdivide the un-annotated attributes into feasible and infeasible sets. Based on that, we can enhance the model to focus on the learning of feasible un-annotated attributes and remove the distraction from the infeasible ones. Besides the internal knowledge, we adopt external knowledge to excavate the unseen object-attribute pairs. Experimental results show that our proposed loss can achieve state-of-the-art performance on the newly cleaned VAW2 dataset that contains 170,407 instances, 1763 objects, and 591 attributes.

![Fig2](kesl_fig2.png)

