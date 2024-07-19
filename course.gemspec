# frozen_string_literal: true

Gem::Specification.new do |spec|
  spec.name          = "course"
  spec.version       = "0.1.0"
  spec.authors       = ["Shotsan"]
  spec.email         = ["shotsan@pm.me"]

  spec.summary       = "A simple theme to create course webpages with Jekyll"
  spec.homepage      = "https://github.com/username/course"
  spec.license       = "MIT"

  spec.files         = Dir['course/*'].select { |f| f.match(%r!^(assets|_data|_layouts|_includes|_sass|LICENSE|README|_config\.yml)!i) }

  
  spec.add_runtime_dependency "jekyll", ">= 3.5", "< 5.0"
  spec.add_runtime_dependency "jekyll-feed", "~> 0.9"
  spec.add_runtime_dependency "jekyll-seo-tag", "~> 2.1"
  spec.add_development_dependency "bundler"

end
