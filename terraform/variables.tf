variable "aws_region" {
  type        = string
  description = "AWS region for all resources."
  default     = "us-east-1"
}

variable "project_name" {
  type        = string
  description = "Project resource prefix."
  default     = "ml-sentiment-platform"
}

variable "allowed_ssh_cidr" {
  type        = string
  description = "CIDR allowed to SSH into EC2."
}

variable "ec2_key_name" {
  type        = string
  description = "Existing EC2 key pair name."
}

variable "instance_type" {
  type        = string
  default     = "t3.medium"
}
