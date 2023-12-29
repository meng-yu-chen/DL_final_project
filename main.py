import transtab
import evaluator
import data_preprocess


def supervised_learning(dataset_name, data_configs):

    # load multiple datasets by passing a list of data names
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(dataname = [dataset_name], dataset_config = data_configs )            #,'credit-approval' 'cylinder-bands' 'credit-g'


    # build transtab classifier model
    model = transtab.build_classifier(categorical_columns = cat_cols, numerical_columns = num_cols, binary_columns = bin_cols)

    # specify training arguments, take validation loss for early stopping
    training_arguments = {
        'lr': 1e-4,
        'batch_size': 32,
        'num_epoch':100,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint',
        'patience': 20
    }

    transtab.train(model, trainset, valset[0], **training_arguments)

    model.load('./checkpoint')

    x_test, y_test = testset[0]
    print(f'x_test type: {type(x_test)}')
    print(f'y_test type: {type(y_test)}')


    ypred = transtab.predict(clf = model, x_test = x_test, y_test = y_test)
    #print(f'ypred: {ypred}')
    
    #  Test AUROC results
    aucroc_result = evaluator.auc_fn(y_test, ypred)

    print(f'AUROC results for dataset {dataset_name}: {aucroc_result}')


def transfer_learning_across_table():

    dataset_names = ['credit-g', 'credit-approval', 'dress-sales', 'cylinder-bands']
    data_configs_set1 = {}
    data_configs_set2 = {}

    # preparing data configs
    for idx in range(len(dataset_names)):

        dataset_name = dataset_names[idx]
        data_config = data_preprocess.prepare_dataset_config(dataset_name, set_num = 1)
        data_configs_set1[dataset_name] = data_config
    
    # pretraining on set1
    # load a dataset and start vanilla supervised training
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(dataname = dataset_names, dataset_config = data_configs_set1, set_num = 1)

    # build transtab classifier model
    model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

    # start training
    training_arguments = {
        'lr': 5e-5,
        'batch_size': 64,
        'num_epoch':100,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./pretrained-ckpt',
        'patience': 80
    }
    

    transtab.train(model, trainset, valset, **training_arguments)

    # fine-tuning on set2
    dataset_name = 'dress-sales'
    data_config = data_preprocess.prepare_dataset_config(dataset_name, set_num = 2)
    data_configs_set2[dataset_name] = data_config

    # now let's load another data and try to leverage the pretrained model for finetuning
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(dataname = [dataset_name], dataset_config = data_configs_set2, set_num = 2)

    # load the pretrained model
    model.load('./pretrained-ckpt')

    # update model's categorical/numerical/binary column dict
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})


    # start fune-tuning
    training_arguments = {
        'lr': 1e-4,
        'batch_size': 32,
        'num_epoch':100,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint',
        'patience': 50
    }

    transtab.train(model, trainset, valset, **training_arguments)

    x_test, y_test = testset[0]
    print(f'x_test type: {type(x_test)}')
    print(f'y_test type: {type(y_test)}')


    ypred = transtab.predict(clf = model, x_test = x_test, y_test = y_test)
    #print(f'ypred: {ypred}')
    
    #  Test AUROC results
    aucroc_result = evaluator.auc_fn(y_test, ypred)

    print(f'AUROC results for transfer learning {dataset_name} set2: {aucroc_result}')
    



def transfer_learning(dataset_names, data_configs):

    print(f'for {dataset_names[0]}')

    # set random seed
    #transtab.random_seed(42)

    # load multiple datasets by passing a list of data names
    #allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    #    = transtab.load_data(dataname = [dataset_names[0]], dataset_config = data_configs[dataset_names[0]] )            #,'credit-approval' 'cylinder-bands' 'credit-g'

    # for set1 training
    # build transtab classifier model
    model = transtab.build_classifier(categorical_columns = cat_cols, numerical_columns = num_cols, binary_columns = bin_cols)

    # specify training arguments, take validation loss for early stopping
    training_arguments = {
        'lr': 1e-4,
        'batch_size': 64,
        'num_epoch':100,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./ckpt/pretrained',
        'patience': 30
    }

    transtab.train(model, trainset, valset[0], **training_arguments)

    # save model
    model.save('./ckpt/pretrained')
    
    print(f'for {dataset_names[1]} fine-tuning')

    # for set2 finetune
    # load multiple datasets by passing a list of data names
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(dataname = [dataset_names[1]], dataset_config = data_configs[dataset_names[1]] )  
    

    # load model
    model.load('./ckpt/pretrained')

    # update model's categorical/numerical/binary column dict
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

    training_arguments = {
        #'lr': 1e-4,
        #'batch_size': 64,
        'num_epoch':50,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint',
        #'patience': 50
    }

    transtab.train(model, trainset, valset[0], **training_arguments)

    x_test, y_test = testset[0]
    print(f'x_test type: {type(x_test)}')
    print(f'y_test type: {type(y_test)}')


    ypred = transtab.predict(clf = model, x_test = x_test, y_test = y_test)
    #print(f'ypred: {ypred}')
    
    #  Test AUROC results
    aucroc_result = evaluator.auc_fn(y_test, ypred)

    print(f'AUROC results for dataset credit-g {dataset_names[1]}: {aucroc_result}')


def constractive_learning(): #dataset_name, data_configs

    # pretraining 'credit-g', 'credit-approval', 'dress-sales', 'cylinder-bands'
    dataset_names = ['credit-approval', 'dress-sales', 'cylinder-bands']
    data_configs = {}

    # preparing data configs
    for idx in range(len(dataset_names)):

        dataset_name = dataset_names[idx]
        data_config = data_preprocess.prepare_dataset_config(dataset_name)
        data_configs[dataset_name] = data_config

    # load multiple datasets by passing a list of data names
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(dataname = dataset_names, dataset_config = data_configs)
    
    partition = 1

    # build contrastive learner, set supervised=True for supervised VPCL
    model, collate_fn = transtab.build_contrastive_learner(
        cat_cols, num_cols, bin_cols,
        supervised=False, # if take supervised CL
        num_partition=partition, # num of column partitions for pos/neg sampling
        overlap_ratio=0.5, # specify the overlap ratio of column partitions during the CL
    )

    # start contrastive pretraining training
    training_arguments = {
        'num_epoch': 50,
        'batch_size':32,
        'lr':1e-4,
        'patience': 50,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./pretrained-checkpoint' # save the pretrained model
    }

    # pass the collate function to the train function
    print(f'start pretraining!')
    transtab.train(model, trainset, valset, collate_fn=collate_fn, **training_arguments)

    # fune-tuning ,'credit-approval'
    data_configs = {}
    dataset_name = 'credit-g'
    data_config = data_preprocess.prepare_dataset_config(dataset_name)
    data_configs[dataset_name] = data_config

    
    # load the pretrained model and finetune on a target dataset
    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
        = transtab.load_data(dataname = [dataset_name], dataset_config = data_configs)

    # build transtab classifier model, and load from the pretrained dir
    model = transtab.build_classifier(checkpoint = './pretrained-checkpoint')

    # update model's categorical/numerical/binary column dict
    model.update({'cat':cat_cols,'num':num_cols,'bin':bin_cols})

    # start fine-tuning
    # fine-tuning
    training_arguments = {
        'lr': 2e-5,
        'batch_size': 32,
        'num_epoch':50,
        'eval_metric':'val_loss',
        'eval_less_is_better':True,
        'output_dir':'./checkpoint',
        'patience': 30
    }
    
    print(f'start finetuning!')
    #model.load('./checkpoint')
    transtab.train(model, trainset, valset[0], **training_arguments)
    
    
    x_test, y_test = testset[0]
    ypred = transtab.predict(clf = model, x_test = x_test, y_test = y_test)
    aucroc_result = evaluator.auc_fn(y_test, ypred)

    print(f'AUROC results for constractive learning {dataset_name}, n = {partition} : {aucroc_result}')


if __name__ == '__main__':


    # for supervised_learning
    #data_configs = {}
    #dataset_name = "dress-sales"          #"dress-sales"  'cylinder-bands' "credit-approval" "credit-g"
    #data_config = data_preprocess.prepare_dataset_config(dataset_name)
    #data_configs[dataset_name] = data_config

    
    #data_configs[dataset_name] = data_config
    #supervised_learning(dataset_name, data_configs)

    # for transfer_learning

    #data_configs = {}

    #dataset_names = ["set1", "set2"] #0, 1 credit-g-

    #for i in range(len(dataset_names)):

    
    #    data_config = data_preprocess.prepare_dataset_config(dataset_names[i])
    
    #    data_configs[dataset_names[i]] = data_config

    #transfer_learning(dataset_names, data_configs)

    #transfer_learning_across_table()
    
    # for constractive learning
    constractive_learning()
    
    


