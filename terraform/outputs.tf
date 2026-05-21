output "ecr_repository_url" {
  value = aws_ecr_repository.app.repository_url
}

output "ec2_public_ip" {
  value = aws_instance.app.public_ip
}

output "model_bucket" {
  value = aws_s3_bucket.models.bucket
}

output "cloudwatch_log_group" {
  value = aws_cloudwatch_log_group.api.name
}
