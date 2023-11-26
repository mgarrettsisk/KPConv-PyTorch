from datasets.SemanticKitti import *
from train_SemanticKitti import SemanticKittiConfig as BaseConfig

if __name__ == '__main__':
    # Bring in original configuration from the model trainer
    original_configuration = BaseConfig()

    # Create a dataset object
    training_dataset = SemanticKittiDataset(original_configuration,
                                            set='training',
                                            balance_classes=True
                                            )

    training_sampler = SemanticKittiSampler(training_dataset)

    training_loader = DataLoader(training_dataset, batch_size=1,
                                 sampler=training_sampler,
                                 collate_fn=SemanticKittiCollate,
                                 num_workers=original_configuration.input_threads,
                                 pin_memory=True)

    training_sampler.calib_max_in(original_configuration,
                                  training_loader,
                                  verbose=True)

    training_sampler.calibration(training_loader,
                                 verbose=True)

