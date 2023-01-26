# variables.tf
variable "access_key" {
  type        = string
  description = "The AWS access key"
}

variable "secret_key" {
  type        = string
  description = "The AWS secret key"
}

variable "region" {
  type        = string
  description = "The AWS region"
  default     = "us-east-1"
}
variable "s3_user_bucket_name" {
  description = "Nom du bucket"
  type        = string
}