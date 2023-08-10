from seal.task.attribute_recognition import InstanceAttributeRecognitionTask


if __name__ == "__main__":
    
    ins_att_task = InstanceAttributeRecognitionTask(d_config="projects/gsl")
    ins_att_task.run()

