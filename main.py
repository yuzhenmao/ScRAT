from sklearn import metrics
from sklearn.metrics import accuracy_score
import scipy.stats as st
from torch.optim import Adam
from utils import *
import argparse
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import train_test_split

from model_baseline import *
from Transformer import TransformerPredictor

from dataloader import *


def _str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


def int_or_float(x):
    try:
        return int(x)
    except ValueError:
        return float(x)


parser = argparse.ArgumentParser(description='scRNA diagnosis')

parser.add_argument('--seed', type=int, default=240)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=3e-3)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument("--task", type=str, default="severity")
parser.add_argument('--emb_dim', type=int, default=128)  # embedding dim
parser.add_argument('--h_dim', type=int, default=128)  # hidden dim of the model
parser.add_argument('--dropout', type=float, default=0.3)  # dropout
parser.add_argument('--layers', type=int, default=1)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument("--train_sample_cells", type=int, default=500,
                    help="number of cells in one sample in train dataset")
parser.add_argument("--test_sample_cells", type=int, default=500,
                    help="number of cells in one sample in test dataset")
parser.add_argument("--train_num_sample", type=int, default=20,
                    help="number of sampled data points in train dataset")
parser.add_argument("--test_num_sample", type=int, default=100,
                    help="number of sampled data points in test dataset")
parser.add_argument('--model', type=str, default='Transformer')
parser.add_argument('--dataset', type=str, default=None)
parser.add_argument('--inter_only', type=_str2bool, default=False)
parser.add_argument('--same_pheno', type=int, default=0)
parser.add_argument('--augment_num', type=int, default=0)
parser.add_argument('--alpha', type=float, default=1.0)
parser.add_argument('--repeat', type=int, default=3)
parser.add_argument('--all', type=int, default=1)
parser.add_argument('--min_size', type=int, default=6000)
parser.add_argument('--n_splits', type=int, default=5)
parser.add_argument('--pca', type=_str2bool, default=True)
parser.add_argument('--mix_type', type=int, default=1)
parser.add_argument('--norm_first', type=_str2bool, default=False)
parser.add_argument('--warmup', type=_str2bool, default=False)
parser.add_argument('--top_k', type=int, default=1)

args = parser.parse_args()

# print("# of GPUs is", torch.cuda.device_count())
print(args)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

patient_summary = {}
stats = {}
stats_id = {}

if args.task == 'haniffa' or args.task == 'combat':
    label_dict = {0: 'Non Covid', 1: 'Covid'}
elif args.task == 'severity':
    label_dict = {0: 'mild', 1: 'severe'}
elif args.task == 'stage':
    label_dict = {0: 'convalescence', 1: 'progression'}


def train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, data_augmented, data):
    dataset_1 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='train')
    dataset_2 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='test')
    dataset_3 = MyDataset(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test, fold='val')
    train_loader = torch.utils.data.DataLoader(dataset_1, batch_size=args.batch_size, shuffle=True,
                                               collate_fn=dataset_1.collate)
    test_loader = torch.utils.data.DataLoader(dataset_2, batch_size=1, shuffle=False, collate_fn=dataset_2.collate)
    valid_loader = torch.utils.data.DataLoader(dataset_3, batch_size=1, shuffle=False, collate_fn=dataset_3.collate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = data_augmented[0].shape[-1]
    output_class = 1

    if args.model == 'Transformer':
        model = TransformerPredictor(input_dim=input_dim, model_dim=args.emb_dim, num_classes=output_class,
                                     num_heads=args.heads, num_layers=args.layers, dropout=args.dropout,
                                     input_dropout=0, pca=args.pca, norm_first=args.norm_first)
    elif args.model == 'feedforward':
        model = FeedForward(input_dim=input_dim, h_dim=args.emb_dim, cl=output_class, dropout=args.dropout)
    elif args.model == 'linear':
        model = Linear_Classfier(input_dim=input_dim, cl=output_class)
    elif args.model == 'scfeed':
        model = scFeedForward(input_dim=input_dim, cl=output_class, model_dim=args.emb_dim, dropout=args.dropout, pca=args.pca)

    model = nn.DataParallel(model)
    model.to(device)
    best_model = model

    print(device)

    ################################################################
    # training and evaluation
    ################################################################
    optimizer = Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.warmup:
        scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.epochs // 10,
                                                                 num_training_steps=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5, last_epoch=-1)
    sigmoid = torch.nn.Sigmoid().to(device)

    max_acc, max_epoch, max_auc, max_loss, max_valid_acc, max_valid_auc = 0, 0, 0, 0, 0, 0
    test_accs, valid_aucs, train_losses, valid_losses, train_accs, test_aucs = [], [0.], [], [], [], []
    best_valid_loss = float("inf")
    wrongs = []
    trigger_times = 0
    patience = 2
    for ep in (range(1, args.epochs + 1)):
        model.train()
        train_loss = []


        for batch in (train_loader):
            x_ = torch.from_numpy(data_augmented[batch[0]]).float().to(device)
            y_ = batch[1].to(device)
            mask_ = batch[3].to(device)

            optimizer.zero_grad()

            out = model(x_, mask_)

            loss = nn.BCELoss()(sigmoid(out), y_)
            loss.backward()

            optimizer.step()
            train_loss.append(loss.item())

        scheduler.step()

        train_loss = sum(train_loss) / len(train_loss)
        train_losses.append(train_loss)

        if ep % 1 == 0:
            valid_loss = []
            model.eval()
            pred = []
            true = []
            with torch.no_grad():
                for batch in (valid_loader):
                    x_ = torch.from_numpy(data[batch[0]]).float().to(device).squeeze(0)
                    y_ = batch[1].int().to(device)

                    out = model(x_)
                    out = sigmoid(out)

                    loss = nn.BCELoss()(out, y_ * torch.ones(out.shape).to(device))
                    valid_loss.append(loss.item())

                    out = out.detach().cpu().numpy()

                    # majority voting
                    f = lambda x: 1 if x > 0.5 else 0
                    func = np.vectorize(f)
                    out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)
                    pred.append(out)
                    y_ = y_.detach().cpu().numpy()
                    true.append(y_)
            # pred = np.concatenate(pred)
            # true = np.concatenate(true)

            valid_loss = sum(valid_loss) / len(valid_loss)
            valid_losses.append(valid_loss)

            if (valid_loss < best_valid_loss):
                best_model = copy.deepcopy(model)
                max_epoch = ep
                best_valid_loss = valid_loss
                max_loss = train_loss

            print("Epoch %d, Train Loss %f, Valid_loss %f" % (ep, train_loss, valid_loss))

            # Early stop
            if (ep > args.epochs - 50) and ep > 1 and (valid_loss > valid_losses[-2]):
                trigger_times += 1
                if trigger_times >= patience:
                    break
            else:
                trigger_times = 0

    best_model.eval()
    pred = []
    test_id = []
    true = []
    wrong = []
    prob = []
    with torch.no_grad():
        for batch in (test_loader):
            x_ = torch.from_numpy(data[batch[0]]).float().to(device).squeeze(0)
            y_ = batch[1].int().numpy()
            id_ = batch[2][0]

            out = best_model(x_)
            out = sigmoid(out)
            out = out.detach().cpu().numpy().reshape(-1)

            # For attention analysis:

            # if args.model == 'Transformer':
            #     attens = best_model.module.get_attention_maps(x_)[-1]
            #     for iter in range(len(attens)):
            #         topK = np.bincount(attens[iter].argsort(-1)[:, :, -args.top_k:].
            #                            cpu().detach().numpy().reshape(-1)).argsort()[-20:][::-1]   # 20 is a 
            #         for idd in id_[iter][topK]:
            #             stats[cell_type_large[idd]] = stats.get(cell_type_large[idd], 0) + 1
            #             stats_id[idd] = stats_id.get(idd, 0) + 1

            y_ = y_[0][0]
            true.append(y_)

            if args.model != 'Transformer':
                prob.append(out[0])
            else:
                prob.append(out.mean())

            # majority voting
            f = lambda x: 1 if x > 0.5 else 0
            func = np.vectorize(f)
            out = np.argmax(np.bincount(func(out).reshape(-1))).reshape(-1)[0]
            pred.append(out)
            test_id.append(patient_id[batch[2][0][0][0]])
            if out != y_:
                wrong.append(patient_id[batch[2][0][0][0]])

    if len(wrongs) == 0:
        wrongs = set(wrong)
    else:
        wrongs = wrongs.intersection(set(wrong))

    test_auc = metrics.roc_auc_score(true, prob)

    test_acc = accuracy_score(true, pred)
    for idx in range(len(pred)):
        print(f"{test_id[idx]} -- true: {label_dict[true[idx]]} -- pred: {label_dict[pred[idx]]}")
    test_accs.append(test_acc)

    cm = confusion_matrix(true, pred).ravel()
    recall = cm[3] / (cm[3] + cm[2])
    precision = cm[3] / (cm[3] + cm[1])
    if (cm[3] + cm[1]) == 0:
        precision = 0

    print("Best performance: Epoch %d, Loss %f, Test ACC %f, Test AUC %f, Test Recall %f, Test Precision %f" % (
    max_epoch, max_loss, test_acc, test_auc, recall, precision))
    print("Confusion Matrix: " + str(cm))
    for w in wrongs:
        v = patient_summary.get(w, 0)
        patient_summary[w] = v + 1

    return test_auc, test_acc, cm, recall, precision


if args.model != 'Transformer':
    args.repeat = 60

if args.task != 'custom':
    p_idx, labels_, cell_type, patient_id, data, cell_type_large = Covid_data(args)
else:
    p_idx, labels_, cell_type, patient_id, data, cell_type_large = Custom_data(args)
rkf = RepeatedKFold(n_splits=abs(args.n_splits), n_repeats=args.repeat * 100, random_state=args.seed)
num = np.arange(len(p_idx))
accuracy, aucs, cms, recalls, precisions = [], [], [], [], []
iter_count = 0

for train_index, test_index in rkf.split(num):
    if args.n_splits < 0:
        temp_idx = train_index
        train_index = test_index
        test_index = temp_idx

    label_stat = []
    for idx in train_index:
        label_stat.append(labels_[p_idx[idx][0]])
    unique, cts = np.unique(label_stat, return_counts=True)
    if len(unique) < 2 or (1 in cts):
        continue
#     print(dict(zip(unique, cts)))

    kk = 0
    while True:
        train_index_, valid_index, ty, vy = train_test_split(train_index, label_stat, test_size=0.33,
                                                             random_state=args.seed + kk)
        if len(set(ty)) == 2 and len(set(vy)) == 2:
            break
        kk += 1

    train_index = train_index_
    len_valid = len(valid_index)
    _index = np.concatenate([valid_index, test_index])

    train_ids = []
    for i in train_index:
        train_ids.append(patient_id[p_idx[i][0]])
#     print(train_ids)

    x_train = []
    x_test = []
    x_valid = []
    y_train = []
    y_valid = []
    y_test = []
    id_train = []
    id_test = []
    id_valid = []
    data_augmented, train_p_idx, labels_augmented, cell_type_augmented = mixups(args, data,
                                                                                [p_idx[idx] for idx in train_index],
                                                                                labels_,
                                                                                cell_type)
    individual_train, individual_test = sampling(args, train_p_idx, [p_idx[idx] for idx in _index], labels_,
                                                 labels_augmented, cell_type_augmented)
    for t in individual_train:
        id, label = [id_l[0] for id_l in t], [id_l[1] for id_l in t]
        x_train += [ii for ii in id]
        y_train += (label)
        id_train += (id)

    temp_idx = np.arange(len(_index))
    for t_idx in temp_idx[len_valid:]:
        id, label = [id_l[0] for id_l in individual_test[t_idx]], [id_l[1] for id_l in individual_test[t_idx]]
        x_test.append([ii for ii in id])
        y_test.append(label[0])
        id_test.append(id)
    for t_idx in temp_idx[:len_valid]:
        id, label = [id_l[0] for id_l in individual_test[t_idx]], [id_l[1] for id_l in individual_test[t_idx]]
        x_valid.append([ii for ii in id])
        y_valid.append(label[0])
        id_valid.append(id)
    x_train, x_valid, x_test, y_train, y_valid, y_test = x_train, x_valid, x_test, np.array(y_train).reshape([-1, 1]), \
                                                         np.array(y_valid).reshape([-1, 1]), np.array(y_test).reshape(
        [-1, 1])
    auc, acc, cm, recall, precision = train(x_train, x_valid, x_test, y_train, y_valid, y_test, id_train, id_test,
                                            data_augmented, data)
    aucs.append(auc)
    accuracy.append(acc)
    cms.append(cm)
    recalls.append(recall)
    precisions.append(precision)
    iter_count += 1
    if iter_count == abs(args.n_splits) * args.repeat:
        break

    del data_augmented

print("="*33)
print("=== Final Evaluation (average across all splits) ===")
print("="*33)

print("Best performance: Test ACC %f,   Test AUC %f,   Test Recall %f,   Test Precision %f" % (np.average(accuracy), np.average(aucs), np.average(recalls), np.average(precisions)))

####################################
######## Only for repeat > 1 #######
####################################
# accuracy = np.array(accuracy).reshape([-1, args.repeat]).mean(0)
# aucs = np.array(aucs).reshape([-1, args.repeat]).mean(0)
# recalls = np.array(recalls).reshape([-1, args.repeat]).mean(0)
# precisions = np.array(precisions).reshape([-1, args.repeat]).mean(0)
# ci_1 = st.t.interval(alpha=0.95, df=len(accuracy) - 1, loc=np.mean(accuracy), scale=st.sem(accuracy))[1] - np.mean(accuracy)
# ci_2 = st.t.interval(alpha=0.95, df=len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))[1] - np.mean(aucs)
# ci_3 = st.t.interval(alpha=0.95, df=len(recalls) - 1, loc=np.mean(recalls), scale=st.sem(recalls))[1] - np.mean(recalls)
# ci_4 = st.t.interval(alpha=0.95, df=len(precisions) - 1, loc=np.mean(precisions), scale=st.sem(precisions))[1] - np.mean(precisions)
# print("ci: ACC ci %f,   AUC ci %f,   Recall ci %f,   Precision ci %f" % (ci_1, ci_2, ci_3, ci_4))

# print(np.average(cms, 0))
# print(patient_summary)
# print(stats)
# print(stats_id)
