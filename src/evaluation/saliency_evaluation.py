import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def generate_saliency_maps(model, image):
  """Generates saliency maps for the given image.

  Args:
    model: The deep learning model.
    image: The input image.

  Returns:
    The saliency maps for the output layer of the model.
  """

  # Get the output of the model.
  output = model(image)

  # Calculate the gradients of the output with respect to the input image.
  gradients = torch.autograd.grad(output, image, create_graph=True)[0]

  # Normalize the gradients.
  gradients = gradients / torch.norm(gradients, p=2, dim=1, keepdim=True)

  # Propagate the gradients back through the model.
  saliency_maps = []
  for layer in model.modules():
    if isinstance(layer, nn.Conv2d):
      gradients = F.conv2d(gradients, layer.weight.transpose(1, 2), stride=layer.stride, padding=layer.padding)
    saliency_maps.append(gradients)

  return saliency_maps

def calculate_integrated_gradients(model, image, baseline):
  """Calculates integrated gradients for the given image and baseline.

  Args:
    model: The deep learning model.
    image: The input image.
    baseline: The baseline image.

  Returns:
    The integrated gradients for the output layer of the model.
  """

  # Get the output of the model for the image and the baseline.
  output = model(image)
  baseline_output = model(baseline)

  # Calculate the gradients of the output with respect to the input image.
  gradients = torch.autograd.grad(output - baseline_output, image, create_graph=True)[0]

  # Integrate the gradients over the input space.
  integrated_gradients = torch.sum(gradients, dim=0)

  return integrated_gradients

def quantify_saliency_maps(saliency_maps, metrics):
  """Quantifies the saliency maps using the given metrics.

  Args:
    saliency_maps: The saliency maps to quantify.
    metrics: The metrics to use.

  Returns:
    A dictionary of metrics for each saliency map.
  """

  metrics_dict = {}
  for saliency_map in saliency_maps:
    metrics_dict[saliency_map] = {}
    for metric in metrics:
      if metric == "magnitude":
        metrics_dict[saliency_map][metric] = torch.mean(saliency_map)
      elif metric == "spatial_distribution":
        metrics_dict[saliency_map][metric] = torch.mean(saliency_map[0:200, :], axis=0)
      elif metric == "consistency":
        metrics_dict[saliency_map][metric] = torch.corrcoef(saliency_map, saliency_map[::2, :])[0, 1]
      else:
        raise ValueError("Unknown metric: {}".format(metric))

  return metrics_dict

def main():
  # Load the model.
  model = torch.load("model.pth")

  # Load the image.
  image = torch.from_numpy(np.array(image)).float()

  # Generate the saliency maps.
  saliency_maps = generate_saliency_maps(model, image)

  # Quantify the saliency maps.
  metrics = ["magnitude", "spatial_distribution", "consistency"]
  metrics_dict = quantify_saliency_maps(saliency_maps, metrics)

  # Print the metrics.
  for saliency_map in saliency_maps:
    print("Saliency map: {}".format(saliency_map))
    for metric in metrics:
      print("  {}: {}".format(metric, metrics_dict[saliency_map][metric]))

if __name__ == "__main__":
  main()
