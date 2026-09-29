import React from 'react';
import { motion } from 'framer-motion';

const AboutSection = () => {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="bg-gray-800/50 backdrop-blur-md rounded-2xl p-6 border border-gray-700/50 shadow-2xl"
    >
      <h3 className="text-xl font-bold text-foreground mb-4">How This Works</h3>
      <p className="text-gray-300 leading-relaxed">
        This project uses a Convolutional Neural Network (CNN) trained on real and fake image datasets.
        Occlusion sensitivity highlights image regions that most influence the model's prediction by
        measuring how the score changes when small areas are hidden.
      </p>
    </motion.div>
  );
};

export default AboutSection;
