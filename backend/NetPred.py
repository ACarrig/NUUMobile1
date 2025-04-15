import model_building.NNet as NN

Main_Model = NN.Churn_Network(init_mode="load_model", args="model_building/MLPCModel")

def predict_churn(file, sheet):
    return Main_Model.Sheet_Predict_default(file, sheet)
