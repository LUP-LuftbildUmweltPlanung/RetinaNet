import rasterio
import shapely
import geopandas
import torch

from deepforest import main


def predict_in_tiles(image_path, output_shape, pre_model=None, multi=False, label_dict=None, gpu=True,
                     score_threshold=0.05, patch_size=400, patch_overlap=0.1):
    """
    Predicts on a single image file using the provided model.

                Keyword arguments
                -----------------
                image_path:         Annotations file to be used during validation
                output_shape:       Path+File_Name for storing the prediction
                pre_model:      Model to be validated
                multi:              If the annotations file contains multiple classes (default=False)
                label_dict:         Label dictionary as created by the model during training
                                    (default=None, only required when training with multiple classes)
                gpu:                If GPU should be used for prediction (default=True)
                score_threshold:    Threshold for considering something a detection (default=0.05)
                patch_size:         Size of resulting crops (square, default=400)
                patch_overlap:      Overlap of resulting crops (default=0.0)
    """

    if pre_model is not None:
        if multi:
            assert label_dict is not None, "Please set a label_dict when using multi-class models"
            m = main.deepforest(num_classes=len(label_dict), label_dict=label_dict)
            if gpu == 1:
                m.to("cuda")
            else:
                m.to("cpu")
            ckpt = torch.load(pre_model, map_location=torch.device("cuda" if gpu else "cpu"))
            m.load_state_dict(ckpt["state_dict"])
        else:
            m = main.deepforest()
            m.load_from_checkpoint(pre_model)
    else:
        m = main.deepforest()
        if gpu == 1:
            m.to("cuda")
        else:
            m.to("cpu")
        print("using standard neon model...")
        m.use_release()

    prediction = m.predict_tile(image_path, patch_size=patch_size, patch_overlap=patch_overlap)

    if prediction is not None:
        m.config["score_threshold"] = score_threshold

        with rasterio.open(image_path) as dataset:
            bounds = dataset.bounds
            pixelSizeX, pixelSizeY = dataset.res

        # subtract origin. Recall that numpy origin is top left! Not bottom left.
        prediction["xmin"] = (prediction["xmin"] * pixelSizeX) + bounds.left
        prediction["xmax"] = (prediction["xmax"] * pixelSizeX) + bounds.left
        prediction["ymin"] = bounds.top - (prediction["ymin"] * pixelSizeY)
        prediction["ymax"] = bounds.top - (prediction["ymax"] * pixelSizeY)

        # combine column to a shapely Box() object, save shapefile
        prediction['geometry'] = prediction.apply(lambda x: shapely.geometry.box(x.xmin, x.ymin, x.xmax, x.ymax),
                                                  axis=1)
        prediction = geopandas.GeoDataFrame(prediction, geometry='geometry')

        # set projection, (see dataset.crs) hard coded here
        if dataset.crs.is_epsg_code:
            code = int(dataset.crs['init'].lstrip('epsg:'))
            prediction.crs = {'init': 'epsg:' + str(code)}
        else:
            print("Dataset did not contain a epsg code. File will be saved without epsg code.")

        # boxes.to_file(r'C:\benny\dops_rgbi_heilige_hallen\\'+dop_name+".shp", driver='ESRI Shapefile')
        prediction.to_file(output_shape, driver='ESRI Shapefile')
    else:
        print("No file saved.")
