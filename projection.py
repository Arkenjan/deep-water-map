import gdal

def set_projection(in_raster, proj_raster):
    dataset1 = gdal.Open(proj_raster)
    projection = dataset1.GetProjection()
    geotransform = dataset1.GetGeoTransform()

    dataset2 = gdal.Open(in_raster, gdal.GA_Update)
    dataset2.SetGeoTransform(geotransform)
    dataset2.SetProjection(projection)


if __name__ == '__main__':

    in_ras = "results/water_map3.tif"
    proj_ras = "sample_data/LC80320272015140LGN01.tif"
    set_projection(in_ras, proj_ras)