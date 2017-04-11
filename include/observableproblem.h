//
// Created by heuer on 11.04.17.
//

#ifndef OBSERVABLEPROBLEM_H
#define OBSERVABLEPROBLEM_H

#include <vector>
#include <iostream>
#include "problemobserver.h"

namespace cppoptlib {
    class ObservableProblem {
    public:
        void addObserver(ProblemObserver *observer) {
            std::cout << "Obersver added: " << observer << std::endl;
            observers_.push_back(observer);
        }

        void removeObserver(ProblemObserver *observer) {
            observers_.erase(std::remove(observers_.begin(), observers_.end(), observer), observers_.end());
        }

        void notifyObserversAboutPerformedStep() {
            std::cout << "step performed" << std::endl;
            for (auto &observer: observers_) observer->stepPerformed();
        }

        unsigned long getObserverCount() {
            return observers_.size();
        }

    private:
        std::vector<ProblemObserver *> observers_;
    };
}

#endif //OBSERVABLEPROBLEM_H
